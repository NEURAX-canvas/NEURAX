"""Agent orchestration - main run loop using the 3-phase declarative pipeline."""
import asyncio
import logging
import os
from typing import Any, Optional

from langchain_runner import select_family
from snapshot_ops import _apply_tool_to_snapshot
from catalogue_store import get_catalogue_for_family, get_all_blocks, get_family_constraints
from suggestions import _rehydrate_catalogue

# New 3-phase modules
from arch_planner import plan_architecture, plan_strategy
from topology_validator import validate_arch_spec, auto_repair_fanin_violations, ArchSpec
from requirements import extract_budget
from budget_check import measure_and_check, narrow_precision_to_fit
from layout_engine import assign_positions
from materializer import materialize

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [NEURAX-AGENT] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def _event(event_type: str, data: Any) -> dict[str, Any]:
    """Helper to format SSE events."""
    return {"event": event_type, "data": data}


async def _run_agent(
    run_id: str,
    q: "asyncio.Queue[dict[str, Any]]",
    user_message: str,
    snapshot: dict[str, Any],
    _runs: dict[str, "asyncio.Queue[dict[str, Any]]"],
    creativity: float = 0.0,
    credentials: dict[str, Any] | None = None,
) -> None:
    """
    Main agent orchestration using the 3-phase declarative pipeline:
    1. Phase 1: Planning (LLM generates a full ArchSpec)
    2. Phase 2: Validation & Layout (Pure Python correctness check + auto-layout)
    3. Phase 3: Materialization (Stream tool calls to the canvas)
    """
    logger.info(f"🚀 AGENT STARTED (3-PHASE) | run_id={run_id} | creativity={creativity}")
    logger.info(f"   User message: {user_message[:100]}...")

    try:
        # ── 0. Setup & Family Selection ──
        _rehydrate_catalogue(snapshot)
        
        current_family = str(snapshot.get("family") or "").strip()
        allowed_families = list(snapshot.get("allowed_families") or [])
        selected_family = current_family

        if allowed_families:
            try:
                selected_family = await select_family(
                    user_message=user_message,
                    allowed_families=allowed_families,
                    catalogue=list(snapshot.get("catalogue") or []),
                    current_family=current_family or None,
                    max_retries=3,
                )
            except Exception as e:
                logger.error(f"❌ Family selection failed: {e}")
                selected_family = current_family or allowed_families[0]

            if selected_family != current_family:
                logger.info(f"🔄 Family changed: {current_family} → {selected_family}")
                await q.put(_event("assistant", {"content": f"I've selected the '{selected_family}' family for this architecture."}))
                # Apply set_family immediately
                tool = {"name": "set_family", "args": {"family": selected_family}}
                await q.put(_event("tool", tool))
                snapshot = _apply_tool_to_snapshot(snapshot, tool)
        
        # Ensure we have the right catalogue for the selected family
        family_catalogue = get_catalogue_for_family(selected_family)
        if not family_catalogue:
            family_catalogue = get_all_blocks()
        
        # Family-specific constraints (from catalogue_store / catalogue.json)
        constraints = get_family_constraints(selected_family)

        # ── Phase 0: Orchestration Planning ──
        logger.info("📋 Phase 0: Generating orchestration strategy...")
        strategy_items = await plan_strategy(
            user_message=user_message,
            family=selected_family,
            hw_config=snapshot.get("hw_config")
        )
        
        # Emit initial plan
        plan_data = [{"id": item.id, "text": item.text, "status": "pending"} for item in strategy_items]
        if plan_data:
            plan_data[0]["status"] = "in_progress"
        await q.put(_event("plan", {"items": plan_data}))

        def _update_plan(idx: int, status: str):
            if idx < len(plan_data):
                plan_data[idx]["status"] = status
                # If we mark one as done, mark next as in_progress
                if status == "done" and idx + 1 < len(plan_data):
                    plan_data[idx+1]["status"] = "in_progress"
            return _event("plan", {"items": plan_data})

        # ── 1. Phase 1: Architecture Planning (Structured LLM Spec) ──
        await q.put(_event("assistant", {"content": "Designing the architecture specification..."}))
        
        spec: Optional[ArchSpec] = None
        validation_result = None
        previous_errors: list[str] = []

        hw_config = snapshot.get("hw_config")

        # What the client actually asked for, read from their own words. A design
        # that is structurally valid but twice the size they allowed is not the
        # model they requested, so the budget is part of the acceptance test.
        budget = extract_budget(user_message, hw_config)
        if not budget.is_empty():
            logger.info(f"🎯 Client budget: {budget.describe()}")
            await q.put(_event("assistant", {
                "content": f"Designing to your stated budget — {budget.describe()}."
            }))

        budget_report = None
        MAX_ATTEMPTS = 4

        for attempt in range(1, MAX_ATTEMPTS + 1):
            logger.info(f"📋 Planning attempt {attempt}/{MAX_ATTEMPTS}...")
            try:
                spec = await plan_architecture(
                    credentials=credentials,
                    user_message=user_message,
                    family=selected_family,
                    catalogue=family_catalogue,
                    constraints=constraints,
                    creativity=creativity,
                    hw_config=hw_config,
                    strategy=[item.text for item in strategy_items],
                    # Read before materialize's clear_canvas wipes it — lets
                    # the planner build on what's already there instead of
                    # guessing a whole architecture from a short edit request.
                    existing_nodes=snapshot.get("nodes"),
                    existing_connections=snapshot.get("connections"),
                    previous_errors=previous_errors if previous_errors else None
                )

                # ── 2. Phase 2: Structural validation ──
                validation_result = validate_arch_spec(spec, family_catalogue, constraints)

                # A fan-in violation (usually several heads converging on one
                # node) has one mechanical fix — insert a merge block — that a
                # graph transform applies deterministically. Try that before
                # spending a planning attempt asking the model to do the same
                # edit, which it does not reliably get right even when told
                # exactly what is wrong.
                if not validation_result.valid:
                    repaired = auto_repair_fanin_violations(spec, family_catalogue)
                    if repaired is not None:
                        repaired_result = validate_arch_spec(repaired, family_catalogue, constraints)
                        if repaired_result.valid:
                            logger.info("🔧 Auto-repaired a fan-in violation without a planning retry")
                            await q.put(_event("assistant", {
                                "content": "Auto-fixed: merged multiple branches before the node that only accepts one input."
                            }))
                            spec = repaired
                            validation_result = repaired_result

                if not validation_result.valid:
                    previous_errors = validation_result.errors
                    logger.warning(f"⚠️ Validation failed on attempt {attempt}: {previous_errors}")
                    if attempt < MAX_ATTEMPTS:
                        await q.put(_event("assistant", {"content": f"Refining design (fix: {previous_errors[0]})..."}))
                    continue

                # ── 2b. Phase 2b: Compile the design ──
                #
                # Always compiled, budget or not. The compiler is the only thing
                # that knows whether a design will actually run, and an analysis
                # costs milliseconds. Skipping it when the client stated no
                # budget — which is most requests — meant the agent delivered
                # designs the compiler had already judged unable to start,
                # without ever asking it.
                stated_budget = not budget.is_empty()
                await q.put(_event("assistant", {
                    "content": "Compiling the design to check it against your budget..."
                    if stated_budget
                    else "Compiling the design to check it holds up..."
                }))
                budget_report = await measure_and_check(spec, budget, hw_config)

                if budget_report.error:
                    # The compiler is the authority on size; without it we ship
                    # the structurally valid design and say the budget is unverified.
                    logger.warning(f"⚠️ Not verified: {budget_report.error}")
                    await q.put(_event("assistant", {
                        "content": f"Could not compile the design ({budget_report.error}); "
                                   "delivering it unmeasured."
                    }))
                    break

                # What the compiler says about the design itself, separately
                # from whether it meets a budget. A model can be comfortably
                # under a size limit and still be one that will not start.
                blocking = budget_report.blocking_diagnostics()
                if blocking:
                    logger.warning(
                        "🚫 Compiler diagnostics on attempt %s: %s",
                        attempt,
                        "; ".join(str(d.get("message", "")) for d in blocking),
                    )
                    for diagnostic in blocking[:5]:
                        await q.put(_event("assistant", {
                            "content": f"The compiler flags this design: {diagnostic.get('message', '')}"
                        }))

                if stated_budget:
                    logger.info("📏 Budget check:\n%s", budget_report.summary())
                    await q.put(_event("assistant", {"content": budget_report.summary()}))

                if budget_report.fits and not blocking:
                    logger.info(f"✅ Design holds up on attempt {attempt}")
                    break

                if blocking and attempt < MAX_ATTEMPTS:
                    # Hand the compiler's own words to the planner and try
                    # again, rather than reaching for the precision lever —
                    # which does nothing for a design that is structurally
                    # wrong.
                    previous_errors = budget_report.planner_feedback()
                    continue

                if budget_report.fits:
                    break

                # Storage width is the one lever that costs nothing to pull: it
                # changes how weights are stored without touching the design the
                # client described. Apply it here and re-measure, rather than
                # asking the planner to and spending an attempt finding out
                # whether it did.
                previous_precision = (
                    (spec.hw_config or {}).get("precision")
                    or (hw_config or {}).get("precision")
                    or "fp16"
                )
                narrowed, narrowed_report = await narrow_precision_to_fit(
                    spec, budget, hw_config, budget_report
                )
                if narrowed:
                    budget_report = narrowed_report
                    logger.info(f"✅ Budget met by narrowing {previous_precision} -> {narrowed}")
                    # A precision change is a design decision the client should
                    # see, not something to slip in silently.
                    await q.put(_event("assistant", {
                        "content": (
                            f"Storing weights in {narrowed} instead of {previous_precision} "
                            f"brings the design inside your budget without changing the "
                            f"architecture.\n\n{budget_report.summary()}"
                        )
                    }))
                    break

                previous_errors = budget_report.planner_feedback()
                if attempt < MAX_ATTEMPTS:
                    await q.put(_event("assistant", {"content": "Over budget — resizing the design..."}))

            except Exception as e:
                logger.error(f"❌ Planning attempt {attempt} failed: {e}")
                if attempt == MAX_ATTEMPTS:
                    raise

        if not validation_result or not validation_result.valid:
            error_details = "; ".join(previous_errors) if previous_errors else "Unknown error"
            raise ValueError(f"Could not generate a valid architecture: {error_details}")

        # Report honestly when the budget could not be met rather than shipping a
        # design that silently misses what the client asked for.
        if budget_report and not budget_report.fits and not budget_report.error:
            over = "; ".join(c.describe() for c in budget_report.checks if not c.fits)
            logger.warning(f"⚠️ Delivering a design that misses the budget: {over}")
            await q.put(_event("assistant", {
                "content": (
                    "I could not reach your budget within the attempts available. "
                    f"The closest design still misses it: {over}. "
                    "Relaxing the budget, or accepting a narrower dtype, would close the gap."
                )
            }))

        if spec is not None and spec.rationale:
            await q.put(_event("assistant", {"content": spec.rationale}))
        await q.put(_update_plan(0, "done"))

        # ── 3. Phase 2e: Layout ──
        logger.info("📐 Computing optimal layout...")
        positions = assign_positions(spec)

        # ── 4. Phase 3: Materialization ──
        logger.info("🔧 streaming tool calls to canvas...")
        # Mark middle steps as done during materialization (simplification)
        for i in range(1, len(plan_data) - 1):
             await q.put(_update_plan(i, "done"))

        count = 0
        async for tool_call in materialize(spec, positions):
            await q.put(_event("tool", tool_call))
            snapshot = _apply_tool_to_snapshot(snapshot, tool_call)
            count += 1
            # Slight delay to make the UI feel "alive" as it builds
            if tool_call["name"] != "clear_canvas":
                await asyncio.sleep(0.05)

        # Mark final step as done
        await q.put(_update_plan(len(plan_data) - 1, "done"))
        logger.info(f"✨ Architecture materialized with {count} tool calls")

    except Exception as e:
        logger.error(f"💥 Agent Run Failed: {e}", exc_info=True)
        await q.put(_event("error", {"message": str(e)}))
        await q.put(_event("assistant", {"content": f"I encountered an error while building the architecture: {str(e)}"}))
    finally:
        await q.put(_event("done", {}))

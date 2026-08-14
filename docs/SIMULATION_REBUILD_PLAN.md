# Simulation Rebuild Plan

An audit of every chart card in the Simulation workspace, verified against the
compiler's actual JSON output and the client code that reads it — not against
the TypeScript types describing what it was supposed to send. Of 47 cards
across 8 tabs, 4 never receive data on any model, 6 more show numbers invented
in the browser, and several duplicate the same field three or four times
across different tabs. This document is the evidence and the plan to fix it.

An interactive version of this audit — the full 47-row schedule with source
citations — is published as an artifact:
https://claude.ai/code/artifact/af689a62-61bc-43e2-aadd-77e09b9c55a3

## What's actually broken

### A — Structurally dead. Four charts, on every model, forever.

Verified against `neurax-ir/src/report/json_output.rs`: `memory_liveness`,
`memory_heatmap`, `live_trace.partial_metrics` and `live_trace.throughput_trace`
are read by the client's `AnalysisResult` type, but **no struct in the
compiler's JSON output defines them**, and no code path in `Index.tsx` ever
assigns them a value, streaming or not.

```ts
// Index.tsx — every one of these four defaults to [] and is never reassigned
memory_liveness: [],      // ← MemoryCharts "VRAM Liveness"
memory_heatmap: [],       // ← MemoryCharts "Memory Heatmap (Timeline)"
live_trace: {
  partial_metrics: [],    // ← RealTimeCharts "Partial Metrics"
  throughput_trace: [],   // ← RealTimeCharts "Instantaneous Throughput"
}
```

Not empty because the analysis hasn't run. Empty because nothing on either
side of the wire ever fills them in. **Verdict: cut, no replacement.**

### B — Only real during a live session

`compilation.phase_timeline` (card: **Phase Timeline**) and the progress
fields behind **Global Progress** arrive only through `/analyze/stream`, the
path behind the **Run Analysis** button. The moment an analysis runs through
the synchronous fallback — which happens automatically after every import —
these two cards go empty, with no indication that the reason is which code
path ran a moment earlier.

**Resolved behavior:** these two stop being tab cards you can revisit. They
become a transient overlay shown only while an analysis is actively streaming
— visible during the run, gone once it completes, never rendered in an
`Absent` or stale state because there is nothing to be stale. The synchronous
path never shows them at all, by construction, rather than showing them empty.

### C — Numbers invented in the browser, shown at the same weight as measured ones

From `simulationData.ts` and `OptimizationCharts.tsx`:

```ts
deriveFusionCandidates()  // "Layer Fusion Candidates"
  if (sourceType.includes('attention') && targetType.includes('ffn'))
    gainPct = 14;  // no profiling, no measurement — a string match

deriveVariantRows()       // "Scaled Cluster" / "INT8 Quant" rows
  speedLabel: '1.35x',  // constant, not computed from the design

CostSummary()             // "Cost Summary (Estimated)" — NOT the same card as
                           // "Cost Breakdown" in Training, which is real
  flopsCost = totalFlops * 2e-11   // invented $/FLOP, ignores the real trainingCostUsd
```

`Layer Fusion Candidates` and `deriveVariantRows`'s multipliers have no
formula behind them at all — a constant or a substring match is not an
estimate of anything, and both are cut outright, not relabeled.

**`Cost Summary (Estimated)` needs the distinction stated precisely, because
its name is one field-name collision away from a real card:** the workspace
has two differently-named cost cards. `Cost Summary (Estimated)` (Overview)
splits an invented per-FLOP, per-byte price across three components that add
up to nothing the compiler actually computed — **cut**. `Cost Breakdown`
(Training & Cost) is a pie of `analysis.trainingCostUsd`, a real field — kept
as-is. The title said "(Estimated)" on the fabricated one, but "estimated"
only applies to a stated formula over real inputs; an invented per-unit price
is neither real nor a formula over something real, so it doesn't qualify for
the `Estimated` state at all — it's `Absent` masquerading as a number. That is
the boundary the `Estimated` state draws: a card earns it only when every
input in its formula traces back to a compiler field.

### D — The same number, drawn four different ways, in four different tabs

`analysis.confidenceScore` is one real field, computed once by the compiler.
It currently has four presentations, and the canonical answer is: **use that
field verbatim, everywhere, and stop recomputing it.**

| Current card | Tab | Disposition |
|---|---|---|
| Confidence Score | Overview | **Kept** — becomes the one confidence card, showing `analysis.confidenceScore` directly |
| Shape Confidence | Diagnostics | **Removed as a separate card.** Its two grounded sub-values (`tensorResolutionRatio`, the unresolved-dimension count) fold into the Overview card as supporting detail; its two heuristic sub-scores ("Diagnostics", "Memory Fit" — blended penalties with no compiler source) are dropped, not relabeled |
| Confidence Score Live | Compilation | **Cut** — a fifth rendering of the same field, in a tab that's becoming a transient overlay |
| Hardware Fit Score | Overview | **Kept, as a distinct metric** — it answers "does this fit the target GPU", not "how sure is the compiler about this shape". It stays separate precisely because averaging it into confidence is what caused the disagreement in the first place |

## The rule going forward

Every card declares which of three states it's in. There is no fourth state,
and nothing is silently promoted out of the one it's actually in.

| State | Meaning |
|---|---|
| **Real** | Reads one field the compiler computed. No badge needed — the number is the metric. |
| **Estimated** | A stated formula whose every input traces to a real compiler field. Badge on the card header, not a caption inside a row. |
| **Absent** | No data for this model. Says why — "no MoE in this design" — not "no data". |

Data honesty is necessary but not sufficient. A correct number in an unreadable
chart still fails the reader. Every surviving card also gets:

- **One palette, reused, not reinvented.** `simulationData.ts` already carries
  a categorical palette checked for contrast and colorblind separation
  (`SIMULATION_COLORS`, `CATEGORICAL_ORDER`) — the rebuild uses it everywhere
  rather than the ad hoc `hsl(var(--chart-N))` and one-off hex values scattered
  across the category files today.
- **Grouping that matches how the question gets asked.** Cards sit next to the
  other cards that answer the same question ("will it fit", "where does the
  cost sit"), not next to whichever card happened to be written in the same
  file.
- **A chart type earns its place same as the data does.** A five-value
  breakdown is a horizontal bar, not a donut fighting for legibility at 90px;
  a trend over layers is a line, not a bar chart pretending to be one.

## The audit — tally

| Verdict | Count |
|---|---|
| Keep | 21 |
| Merge (redundant, consolidate to one) | 7 |
| Rebuild (real data, formula or placement needs honesty) | 5 |
| Cut (fabricated or structurally dead) | 14 |

Total: 47. The full row-by-row breakdown, grouped by current tab with source
citations, is in the artifact linked above.

**How 47 becomes ~26 tab cards**, derived rather than eyeballed:

- 21 `Keep` cards carry over 1:1.
- The 7 `Merge` cards collapse into **3** net new cards: the three
  op-distribution copies (`FLOPs by Op Type`, `Dialect Distribution`,
  `OpKind Distribution`) become one card in Layers; the confidence trio
  becomes the one Overview card described above; `Parallelism Efficiency`
  relocates to Training & Cost as a single card. `Compute vs Memory Bound`
  folds into the existing Roofline card rather than staying separate, and the
  three Comparison-tab cards (`Hardware Configurations to Compare`,
  `Comparison Results`, `Visual Comparison`) become functionality of the
  already-shipped Compare panel — zero new Simulation cards for that group.
- Of the 5 `Rebuild` cards, 2 (`Global Progress`, `Phase Timeline`) leave the
  tab-card count entirely and become the transient overlay described in
  finding B. `Shape Confidence`'s rebuild is absorbed into the confidence
  merge above rather than staying standalone. That leaves 2 standalone
  rebuilt cards: `Layer Efficiency Score`, `Optimization Opportunities`.

21 + 3 + 2 = **26 persistent tab cards**, plus the 2-card transient overlay
that only exists during an active streamed analysis.

## Five tabs, and one reused panel

| Tab | Content |
|---|---|
| **Overview** | Size, cost, one confidence score, hardware fit |
| **Layers** | FLOPs, params, VRAM per layer, roofline |
| **Memory** | Peak breakdown, fragmentation, gradients, KV cache |
| **Training & Cost** | Cost, runtime, energy, carbon, GPU scaling, parallelism |
| **Diagnostics** | Severity, by-layer, unsupported ops, real recommendations |

That is the whole tab bar — five tabs. **Compare is not a sixth tab.** It's
the A/B panel already shipped elsewhere in the app; the three Comparison-tab
cards being merged (above) become inputs to that existing panel instead of a
duplicate comparison surface living inside Simulation.

Cut entirely as a persistent destination: Compilation (its two real cards
become the transient overlay from finding B, not a tab to revisit), and the
standalone Comparison tab (folded into the Compare panel as just described).

## Rollout

1. **Cut the dead and the fabricated.** Remove the 4 structurally-dead cards
   and the cards with no formula behind them at all (`Layer Fusion
   Candidates`, `deriveVariantRows`'s fixed multipliers, `Cost Summary
   (Estimated)`) — 8 cards total from findings A and C. → zero permanently-blank
   or invented-number charts.
2. **Consolidate.** Collapse 8 tabs to 5 plus the reused Compare panel, per the
   merge groups above. → 47 cards → 26 tab cards + a 2-card transient overlay,
   none redundant.
3. **Rebuild the remaining 2 standalone cards and the confidence merge.**
   `Layer Efficiency Score` and `Optimization Opportunities` get their formula
   stated on the card (and `Optimization Opportunities` drops its fixed-score
   fallback when the compiler sends no real recommendations, showing `Absent`
   instead); the confidence merge follows the disposition table in finding D.
   → every remaining badge is true.
4. **Verify against the running compiler, per state and per path.** One test
   per `Real` card asserts its number against the matching field from a live
   `neurax-service` response, the way the HuggingFace importer already is.
   `Estimated` cards get a second assertion that the badge is present and the
   formula's inputs are all real fields. `Absent` cards get a fixture with no
   grounds for the value (a non-MoE design, for the MoE-only cards) and assert
   the explanatory state, not a blank chart. The two overlay cards get one test
   through `/analyze/stream` and one through the synchronous path, asserting
   the overlay is absent (not empty) on the latter. → a badge lying about its
   source, or a card that shows nothing where it owes a reason, fails CI
   rather than a user's trust.

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
side of the wire ever fills them in.

### B — Only real during a live session

`compilation.phase_timeline` arrives only through `/analyze/stream`, the path
behind the **Run Analysis** button. The moment an analysis runs through the
synchronous fallback — which happens automatically after every import — the
same tab shows the same cards empty, with no indication that the reason is
which code path ran five minutes earlier.

### C — Numbers invented in the browser, shown at the same weight as measured ones

From `simulationData.ts` and `OptimizationCharts.tsx`:

```ts
deriveFusionCandidates()  // "Layer Fusion Candidates"
  if (sourceType.includes('attention') && targetType.includes('ffn'))
    gainPct = 14;  // no profiling, no measurement — a string match

deriveVariantRows()       // "Scaled Cluster" / "INT8 Quant" rows
  speedLabel: '1.35x',  // constant, not computed from the design

CostSummary()             // "Cost Summary (Estimated)"
  flopsCost = totalFlops * 2e-11   // invented $/FLOP, ignores the real trainingCostUsd
```

The title says "(Estimated)" on one of these three. The other two carry no
signal at all — `Layer Fusion Candidates` has no badge on its header, only a
five-word caption buried inside each row.

### D — The same number, drawn four different ways, in four different tabs

`analysis.confidenceScore` is one real field. It is rendered as **Confidence
Score** (Overview), inside **Shape Confidence**'s five-bar breakdown
(Diagnostics), as **Confidence Score Live** (Compilation), and folded into
**Hardware Fit Score**'s average (Overview) — with the Diagnostics version
mixing in two additional sub-scores that are themselves heuristic. A reader
comparing two of these expects them to agree; they don't, because they aren't
the same formula.

## The rule going forward

Every card declares which of three states it's in. There is no fourth state,
and nothing is silently promoted out of the one it's actually in.

| State | Meaning |
|---|---|
| **Real** | Reads one field the compiler computed. No badge needed — the number is the metric. |
| **Estimated** | A stated formula over real inputs. Badge on the card header, not a caption inside a row. |
| **Absent** | No data for this model. Says why — "no MoE in this design" — not "no data". |

## The audit — tally

| Verdict | Count |
|---|---|
| Keep | 19 |
| Merge (redundant, consolidate to one) | 10 |
| Rebuild (real data, formula needs honesty) | 4 |
| Cut (fabricated or structurally dead) | 14 |

The full 47-row breakdown, grouped by current tab with source citations, is
in the artifact linked above.

## Five tabs, and where things land

| Tab | Content |
|---|---|
| **Overview** | Size, cost, one confidence score, hardware fit |
| **Layers** | FLOPs, params, VRAM per layer, roofline |
| **Memory** | Peak breakdown, fragmentation, gradients, KV cache |
| **Training & Cost** | Cost, runtime, energy, carbon, GPU scaling |
| **Diagnostics** | Severity, by-layer, unsupported ops, real recommendations |
| **Compare** *(reused)* | Hardware & A/B — the panel already shipped, not duplicated |

Cut entirely: Compilation as a persistent tab (its two real cards become a
transient overlay during Run Analysis, not something to revisit later), and
the standalone Comparison tab (folded into the Compare A/B panel already
built).

## Rollout

1. **Cut the dead and the fabricated.** Remove the 4 structurally-dead cards
   and the 6 fabricated ones outright. No replacement needed — they were
   never data. → zero permanently-blank charts.
2. **Consolidate.** Collapse 8 tabs to 5. One confidence card instead of four.
   One op-distribution chart instead of three. One comparison system, reusing
   the Compare A/B panel. → 47 cards → ~24, none redundant.
3. **Rebuild the four REBUILD-tagged cards.** Layer Efficiency Score, Shape
   Confidence, Optimization Opportunities, and the SSE progress pair get real
   formulas stated on the card, or move to a transient overlay where that's
   what they actually are. → every remaining badge is true.
4. **Verify against the running compiler.** One test per card, mirroring how
   the HuggingFace importer was checked: run a real design through
   `neurax-service`, assert the chart's number matches the field it claims to
   read. → a badge lying about its source fails CI, not a user's trust.

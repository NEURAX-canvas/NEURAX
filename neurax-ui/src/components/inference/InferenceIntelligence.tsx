import { useState, useEffect, useRef, useCallback } from 'react';
import { InferenceControls, buildDefaultInferenceParams } from './InferenceControls.tsx';
import { BehaviorDashboard } from './BehaviorDashboard.tsx';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { InferenceParams, InferenceReport, simulateInference } from '@/services/neuraxApi.ts';
import { useHardware } from '@/contexts/HardwareContext.tsx';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';
import type { CanvasNode, Connection } from '@/types/architecture.ts';

interface InferenceIntelligenceProps {
  architectureType: ArchitectureFamily;
  /** Blocks on the canvas, so the simulation runs on the real design. */
  nodes?: CanvasNode[];
  connections?: Connection[];
}

export function InferenceIntelligence({
  architectureType,
  nodes = [],
  connections = [],
}: InferenceIntelligenceProps) {
  const { config: design } = useHardware();
  const [params, setParams] = useState<InferenceParams>(() =>
    buildDefaultInferenceParams(architectureType, design),
  );
  const [report, setReport] = useState<InferenceReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runSimulation = useCallback(async (p: InferenceParams) => {
    setLoading(true);
    setError(null);
    try {
      // Send the design when there is one, so the report describes this model
      // rather than an assumed default.
      const topology =
        nodes.length > 0
          ? (compileToNeuraxIR(nodes, connections, {
              modelName: 'NeuraxModel',
              family: architectureType,
              ...design,
            }) as unknown as Record<string, unknown>)
          : undefined;
      const res = await simulateInference({ params: p, ...(topology && { topology }) });
      setReport(res.report);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  }, [nodes, connections, architectureType, design]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => runSimulation(params), 800);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [params, runSimulation]);

  // Re-seed when the design changes, so the simulation follows the model on the
  // canvas rather than the one it was opened with.
  useEffect(() => {
    setParams(buildDefaultInferenceParams(architectureType, design));
  }, [architectureType, design.seqLen, design.precision, design.kvHeads, design.numHeads, design.useCache]);

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden bg-background">
      <InferenceControls
        architectureType={architectureType}
        params={params}
        onParamsChange={setParams}
      />
      <BehaviorDashboard
        architectureType={architectureType}
        report={report}
        loading={loading}
        error={error}
      />
    </div>
  );
}

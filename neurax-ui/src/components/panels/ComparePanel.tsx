import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog.tsx';
import { Button } from '@/components/ui/button.tsx';
import { ArrowRight, Camera, AlertTriangle, GitCompare } from 'lucide-react';
import {
  DesignVariant,
  compareDesigns,
  comparisonGroups,
  formatMetric,
  formatDelta,
} from '@/utils/designComparison.ts';

interface ComparePanelProps {
  isOpen: boolean;
  onClose: () => void;
  /** The design captured earlier, or null if none has been. */
  baseline: DesignVariant | null;
  /** The design on the canvas now, or null if it has not been analysed. */
  candidate: DesignVariant | null;
  /** Replace the baseline with what is on the canvas now. */
  onCapture: () => void;
  /** Forget the baseline. */
  onClear: () => void;
}

/** Colour a change by whether it is an improvement, not by its sign. */
const VERDICT_CLASS: Record<string, string> = {
  better: 'text-emerald-600 dark:text-emerald-400',
  worse: 'text-rose-600 dark:text-rose-400',
  neutral: 'text-muted-foreground',
};

export function ComparePanel({
  isOpen,
  onClose,
  baseline,
  candidate,
  onCapture,
  onClear,
}: ComparePanelProps) {
  const report = baseline && candidate ? compareDesigns(baseline, candidate) : null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-primary" />
            Compare designs
          </DialogTitle>
          <DialogDescription>
            Capture a design as the baseline, change it, and see what moved.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto py-2">
          {!baseline && (
            <div className="rounded-lg border border-dashed border-border p-8 text-center">
              <Camera className="w-8 h-8 mx-auto text-muted-foreground mb-3" />
              <p className="font-medium">No baseline captured</p>
              <p className="text-sm text-muted-foreground mt-1 max-w-sm mx-auto">
                Capture the design on the canvas as a baseline. Edit it afterwards and this
                panel shows both side by side.
              </p>
              <Button className="mt-4" onClick={onCapture} disabled={!candidate}>
                <Camera className="w-4 h-4 mr-1.5" />
                Capture current design
              </Button>
              {!candidate && (
                <p className="text-xs text-muted-foreground mt-2">
                  Run an analysis first — there are no numbers to capture yet.
                </p>
              )}
            </div>
          )}

          {baseline && !candidate && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm">
              <p className="font-medium">Baseline captured: {baseline.name}</p>
              <p className="text-muted-foreground mt-1">
                Run an analysis on the current design to compare against it.
              </p>
            </div>
          )}

          {report && (
            <div className="space-y-4">
              {/* Which two designs these are. */}
              <div className="flex items-center gap-3 text-sm">
                <div className="flex-1 min-w-0 rounded-md border border-border bg-secondary/30 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Baseline
                  </div>
                  <div className="font-medium truncate">{report.baseline.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {report.baseline.blockCount} blocks ·{' '}
                    {new Date(report.baseline.capturedAt).toLocaleTimeString()}
                  </div>
                </div>

                <ArrowRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />

                <div className="flex-1 min-w-0 rounded-md border border-primary/40 bg-primary/10 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Current
                  </div>
                  <div className="font-medium truncate">{report.candidate.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {report.candidate.blockCount} blocks
                  </div>
                </div>
              </div>

              {/* Anything that makes this not an architecture comparison. */}
              {report.incomparable.length > 0 && (
                <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
                  <p className="text-xs font-medium text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    These were analysed under different terms
                  </p>
                  <ul className="text-xs text-muted-foreground mt-1.5 space-y-0.5">
                    {report.incomparable.map((note) => (
                      <li key={note}>• {note}</li>
                    ))}
                  </ul>
                  <p className="text-xs text-muted-foreground mt-1.5">
                    Some of the difference below comes from that, not from the architecture.
                  </p>
                </div>
              )}

              {/* The metrics. */}
              {comparisonGroups(report).map((group) => (
                <div key={group}>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">
                    {group}
                  </h3>
                  <div className="rounded-md border border-border overflow-hidden">
                    <table className="w-full text-sm">
                      {/* Named columns: without them a screen reader reads four
                          unlabelled cells per row and the two numbers are
                          indistinguishable. */}
                      <thead className="sr-only">
                        <tr>
                          <th scope="col">Metric</th>
                          <th scope="col">Baseline</th>
                          <th scope="col">Current</th>
                          <th scope="col">Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.metrics
                          .filter((m) => m.group === group)
                          .map((m, i) => (
                            <tr
                              key={m.key}
                              className={i % 2 ? 'bg-secondary/20' : undefined}
                            >
                              <th
                                scope="row"
                                className="px-3 py-1.5 text-left font-normal text-muted-foreground w-[38%]"
                              >
                                {m.label}
                              </th>
                              <td className="px-3 py-1.5 text-right tabular-nums">
                                {formatMetric(m.baseline, m.unit)}
                              </td>
                              <td className="px-3 py-1.5 text-right tabular-nums font-medium">
                                {formatMetric(m.candidate, m.unit)}
                              </td>
                              <td
                                className={`px-3 py-1.5 text-right tabular-nums text-xs w-[18%] ${VERDICT_CLASS[m.verdict]}`}
                              >
                                {formatDelta(m)}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          {baseline && (
            <Button variant="ghost" onClick={onClear}>
              Clear baseline
            </Button>
          )}
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          {baseline && (
            <Button onClick={onCapture} disabled={!candidate}>
              <Camera className="w-4 h-4 mr-1.5" />
              Make current the baseline
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

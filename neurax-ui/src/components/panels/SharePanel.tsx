import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Checkbox } from '@/components/ui/checkbox.tsx';
import { Share2, Copy, Check, Download, AlertTriangle, Link as LinkIcon } from 'lucide-react';
import { cn } from '@/lib/utils.ts';
import { createShare, shareDownloadUrl, type ShareMode } from '@/services/neuraxApi.ts';
import type { AnalysisResult, CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { useToast } from '@/hooks/use-toast.ts';

interface SharePanelProps {
  isOpen: boolean;
  onClose: () => void;
  nodes: CanvasNode[];
  connections: Connection[];
  groups: NodeGroup[];
  selectedArchitecture: ArchitectureFamily;
  // Only a real, completed analysis is worth publishing — the zeroed
  // placeholder shown before the first run would just publish a report full
  // of zeros. `generatedAt` is only ever set by a real analysis result, same
  // guard `ExportPanel` uses.
  analysisResult: AnalysisResult | null;
}

/** Where a share's `edit_token` is kept so the same browser can delete it
 * later without the user having to copy it down by hand. Best-effort only —
 * a different browser or a cleared cache genuinely loses the ability to
 * delete, which is why the token is also shown once at creation time. */
function rememberEditToken(id: string, editToken: string) {
  try {
    localStorage.setItem(`neurax-share-edit-token:${id}`, editToken);
  } catch {
    // Private browsing / storage disabled — the link still works, deleting
    // it later just needs the token shown on screen right now instead.
  }
}

export function SharePanel({
  isOpen,
  onClose,
  nodes,
  connections,
  groups,
  selectedArchitecture,
  analysisResult,
}: SharePanelProps) {
  const { toast } = useToast();
  const [mode, setMode] = useState<ShareMode>('card');
  const [confirmedFull, setConfirmedFull] = useState(false);
  const [displayName, setDisplayName] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [result, setResult] = useState<{ id: string; url: string } | null>(null);
  const [copied, setCopied] = useState(false);

  const reset = () => {
    setMode('card');
    setConfirmedFull(false);
    setDisplayName('');
    setResult(null);
    setCopied(false);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const canSubmit =
    !!analysisResult && displayName.trim().length > 0 && (mode === 'card' || confirmedFull);

  const handleCreate = async () => {
    if (!analysisResult || !canSubmit) return;
    setIsCreating(true);
    try {
      const created = await createShare({
        mode,
        displayName: displayName.trim(),
        family: selectedArchitecture,
        report: analysisResult as unknown as Record<string, unknown>,
        design: mode === 'full' ? { nodes, connections, groups } : null,
      });
      rememberEditToken(created.id, created.editToken);
      setResult({ id: created.id, url: created.url });
    } catch (err) {
      toast({
        title: 'Could not create the link',
        description: err instanceof Error ? err.message : 'The request failed.',
        variant: 'destructive',
      });
    } finally {
      setIsCreating(false);
    }
  };

  const handleCopy = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(result.url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Share2 className="w-5 h-5 text-primary" />
            Share this analysis
          </DialogTitle>
          <DialogDescription>
            A public link anyone can open — no account needed to create or view it.
          </DialogDescription>
        </DialogHeader>

        {!analysisResult ? (
          <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
            Run analysis first — there is nothing to share yet.
          </div>
        ) : result ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2 rounded-md border border-border bg-secondary/30 px-3 py-2">
              <LinkIcon className="w-4 h-4 text-muted-foreground shrink-0" />
              <code className="flex-1 truncate text-sm">{result.url}</code>
              <Button size="sm" variant="ghost" onClick={handleCopy}>
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </Button>
            </div>
            <a
              href={shareDownloadUrl(result.id)}
              className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
            >
              <Download className="w-3.5 h-3.5" />
              Download the report as JSON
            </a>
            <p className="text-xs text-muted-foreground">
              This link stays live until you delete it — remembered on this device for that, but
              anyone with the link can open it as many times as they like.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <Label htmlFor="share-display-name">Name shown on the link</Label>
              <Input
                id="share-display-name"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="e.g. 7B dense transformer, H100 x8"
                className="mt-1.5"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Shown publicly — not the document's own name, so nothing internal leaks by
                accident.
              </p>
            </div>

            <div className="space-y-2">
              <button
                type="button"
                onClick={() => setMode('card')}
                className={cn(
                  'w-full text-left rounded-lg border p-3 transition-colors',
                  mode === 'card' ? 'border-primary bg-primary/5' : 'border-border hover:bg-secondary/50',
                )}
              >
                <div className="font-medium text-sm">Card — numbers only</div>
                <div className="text-xs text-muted-foreground mt-0.5">
                  Params, FLOPs, memory, cost, energy. No architecture topology.
                </div>
              </button>
              <button
                type="button"
                onClick={() => setMode('full')}
                className={cn(
                  'w-full text-left rounded-lg border p-3 transition-colors',
                  mode === 'full' ? 'border-primary bg-primary/5' : 'border-border hover:bg-secondary/50',
                )}
              >
                <div className="font-medium text-sm">Full design — remixable</div>
                <div className="text-xs text-muted-foreground mt-0.5">
                  Everything in Card, plus the node graph, so a viewer can open and edit it in
                  NEURAX.
                </div>
              </button>
            </div>

            {mode === 'full' && (
              <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
                <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                <div className="flex-1 space-y-2">
                  <p className="text-xs text-amber-900 dark:text-amber-200">
                    This includes your full architecture — every block and connection, visible to
                    anyone with the link. Node and group names are scrubbed automatically; numeric
                    parameters are not.
                  </p>
                  <label className="flex items-center gap-2 text-xs font-medium text-amber-900 dark:text-amber-200">
                    <Checkbox
                      checked={confirmedFull}
                      onCheckedChange={(v) => setConfirmedFull(v === true)}
                    />
                    I understand this publishes the full design
                  </label>
                </div>
              </div>
            )}
          </div>
        )}

        <DialogFooter>
          {result ? (
            <Button onClick={handleClose}>Done</Button>
          ) : (
            <>
              <Button variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              <Button onClick={handleCreate} disabled={!canSubmit || isCreating}>
                {isCreating ? 'Creating link…' : 'Create link'}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

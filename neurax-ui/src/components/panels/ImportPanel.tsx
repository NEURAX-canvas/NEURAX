import { useMemo, useState } from 'react';
import { Upload, FileJson, AlertCircle, CheckCircle2, Copy, Info } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Textarea } from '@/components/ui/textarea.tsx';
import { Label } from '@/components/ui/label.tsx';
import { sampleTransformerJSON, ImportResult } from '@/utils/architectureImporter.ts';
import { parseModelJSON, detectImportSource, DetectedImport } from '@/utils/modelImport.ts';
import { openTextFile } from '@/services/desktopRuntime.ts';
import { useToast } from '@/hooks/use-toast.ts';

interface ImportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onImport: (result: ImportResult) => void;
}

/** A LLaMA-shaped config, short enough to read and real enough to try. */
const SAMPLE_HF_CONFIG = `{
  "architectures": ["MistralForCausalLM"],
  "model_type": "mistral",
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 32768,
  "vocab_size": 32000,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "hidden_act": "silu",
  "tie_word_embeddings": false
}`;

const SOURCE_LABEL: Record<string, string> = {
  huggingface: 'HuggingFace config.json',
  neurax: 'NEURAX design',
};

export function ImportPanel({ isOpen, onClose, onImport }: ImportPanelProps) {
  const [jsonInput, setJsonInput] = useState('');
  const [fileName, setFileName] = useState<string | undefined>(undefined);
  const [previewResult, setPreviewResult] = useState<DetectedImport | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const { toast } = useToast();

  // Shown live under the textarea, before validation, so the user knows the
  // dialog recognised the file rather than wondering which parser will run.
  //
  // Memoised because detection parses the whole text: this box also accepts a
  // NEURAX design export, which is far larger than a config, and re-parsing it
  // on every keystroke makes typing stutter.
  const liveSource = useMemo(
    () => (jsonInput.trim() ? detectImportSource(jsonInput) : 'unknown'),
    [jsonInput],
  );

  const handleValidate = () => {
    setIsValidating(true);

    setTimeout(() => {
      const result = parseModelJSON(jsonInput, fileName);
      setPreviewResult(result);
      setIsValidating(false);

      if (result.error) {
        toast({
          title: 'Could not read this file',
          description: result.error,
          variant: 'destructive',
        });
      } else {
        toast({
          title: 'File read',
          description: `Found ${result.nodes.length} nodes and ${result.connections.length} connections`,
        });
      }
    }, 300);
  };

  const handleImport = () => {
    if (!previewResult || previewResult.error) {
      toast({
        title: 'Cannot Import',
        description: 'Press “Read File” first.',
        variant: 'destructive',
      });
      return;
    }

    onImport(previewResult);
    handleClose();

    toast({
      title: 'Model imported',
      description: `Imported "${previewResult.modelName}" with ${previewResult.nodes.length} nodes`,
    });
  };

  const handleClose = () => {
    setJsonInput('');
    setFileName(undefined);
    setPreviewResult(null);
    onClose();
  };

  const handleLoadSample = (which: 'hf' | 'neurax') => {
    setJsonInput(which === 'hf' ? SAMPLE_HF_CONFIG : sampleTransformerJSON);
    setFileName(undefined);
    setPreviewResult(null);
  };

  /**
   * Pick a file through the host's own dialog.
   *
   * On the desktop this is the native picker, which can reach anywhere on disk
   * — including a model directory cloned from the Hub. In a browser it falls
   * back to a file input.
   */
  const handleOpenFile = async () => {
    let picked;
    try {
      // Reading only: importing a config.json out of a model directory must
      // not leave that file overwritable. See `openTextFile`.
      picked = await openTextFile(['json']);
    } catch (err) {
      // The native side rejects a file it cannot decode — a `config.json` that
      // is not valid UTF-8, or one whose permissions changed since the dialog
      // listed it. Silence here looks like the dialog did nothing.
      toast({
        title: 'Could not read that file',
        description: String(err),
        variant: 'destructive',
      });
      return;
    }
    if (!picked) return;
    setJsonInput(picked.contents);
    setFileName(picked.name);
    setPreviewResult(null);
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileJson className="w-5 h-5 text-primary" />
            Import a model
          </DialogTitle>
          <DialogDescription>
            Open a HuggingFace <code className="text-xs">config.json</code> or a NEURAX design.
            The format is detected from the file.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-4 py-4">
          {/* File picker and samples */}
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="secondary" size="sm" onClick={() => void handleOpenFile()}>
              <Upload className="w-4 h-4 mr-1.5" />
              Open File
            </Button>

            <Button variant="outline" size="sm" onClick={() => handleLoadSample('hf')}>
              <Copy className="w-4 h-4 mr-1.5" />
              Sample config.json
            </Button>

            <Button variant="outline" size="sm" onClick={() => handleLoadSample('neurax')}>
              <Copy className="w-4 h-4 mr-1.5" />
              Sample NEURAX
            </Button>
          </div>

          {/* JSON Input */}
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <Label htmlFor="json-input">
                {fileName ? fileName : 'Model file'}
              </Label>
              {liveSource !== 'unknown' && (
                <Badge variant="outline" className="text-[10px]">
                  {SOURCE_LABEL[liveSource]}
                </Badge>
              )}
            </div>
            <Textarea
              id="json-input"
              value={jsonInput}
              onChange={(e) => {
                setJsonInput(e.target.value);
                setFileName(undefined);
                setPreviewResult(null);
              }}
              placeholder={
                'Paste a HuggingFace config.json here — the one beside the weights in any Hub repository:\n\n' +
                '{\n  "model_type": "llama",\n  "hidden_size": 4096,\n  "num_hidden_layers": 32,\n' +
                '  "num_attention_heads": 32,\n  "num_key_value_heads": 8,\n  "intermediate_size": 11008,\n' +
                '  "vocab_size": 32000\n}'
              }
              className="min-h-[220px] font-mono text-xs"
            />
          </div>

          {/* Preview Result */}
          {previewResult && (
            <div
              className={`p-4 rounded-lg border ${previewResult.error
                  ? 'bg-destructive/10 border-destructive/30'
                  : 'bg-primary/10 border-primary/30'
                }`}
            >
              <div className="flex items-start gap-3">
                {previewResult.error ? (
                  <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1 min-w-0">
                  {previewResult.error ? (
                    <>
                      <p className="font-medium text-destructive">Cannot import this file</p>
                      <p className="text-sm text-muted-foreground mt-1">{previewResult.error}</p>
                    </>
                  ) : (
                    <>
                      <p className="font-medium text-foreground flex items-center gap-2 flex-wrap">
                        {previewResult.modelName}
                        {previewResult.detail && (
                          <Badge variant="secondary" className="text-[10px] font-normal">
                            {previewResult.detail}
                          </Badge>
                        )}
                      </p>

                      {/* What the importer read out of the file. */}
                      {previewResult.notes.length > 0 && (
                        <ul className="text-sm text-muted-foreground mt-2 space-y-0.5">
                          {previewResult.notes.map((note, i) => (
                            <li key={i}>• {note}</li>
                          ))}
                        </ul>
                      )}

                      <p className="text-sm text-muted-foreground mt-2">
                        • {previewResult.nodes.length} blocks, {previewResult.connections.length} connections
                      </p>

                      {/* Anything the file did not say, and this had to invent. */}
                      {previewResult.assumptions.length > 0 && (
                        <div className="mt-3 rounded-md border border-amber-500/30 bg-amber-500/10 p-2.5">
                          <p className="text-xs font-medium text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                            <Info className="w-3.5 h-3.5" />
                            Not stated in the file — assumed
                          </p>
                          <ul className="text-xs text-muted-foreground mt-1 space-y-0.5">
                            {previewResult.assumptions.map((note, i) => (
                              <li key={i}>• {note}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            variant="secondary"
            onClick={handleValidate}
            disabled={!jsonInput.trim() || isValidating}
          >
            {isValidating ? 'Reading...' : 'Read File'}
          </Button>
          <Button
            onClick={handleImport}
            disabled={!previewResult || !!previewResult.error}
          >
            Import
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

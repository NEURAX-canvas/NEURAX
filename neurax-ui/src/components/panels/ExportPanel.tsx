import { useState } from 'react';
import {
  Download,
  FileJson,
  FileText,
  Code,
  Cog,
  Image,
  Box,
  Check,
  Copy,
  Zap,
  Server,
  Network,
  Github,
  Loader2,
  FileCode2,
  AlertTriangle,
  FolderTree,
} from 'lucide-react';

import { cn } from '@/lib/utils.ts';
import { toToml } from '@/utils/tomlExport.ts';
import { saveTextFile } from '@/services/desktopRuntime.ts';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Label } from '@/components/ui/label.tsx';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { CanvasNode, Connection, NodeGroup, AnalysisResult } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';
import { useHardware } from '@/contexts/HardwareContext.tsx';
import { GitHubExportPanel } from './GitHubExportPanel.tsx';
import { ExportAssistant } from './ExportAssistant.tsx';
import { buildProjectFiles, downloadProjectZip, ProjectExportResult } from '@/utils/projectExport.ts';

const iconMap: Record<string, React.ElementType> = {
  FileJson,
  FileText,
  Code,
  Cog,
  Image,
  Box,
  Zap,
  Server,
  Network,
  Github,
  FileCode2,
};

interface ExportOption {
  id: string;
  name: string;
  description: string;
  extension: string;
  icon: string;
  includeAnalysis?: boolean;
  minPlan: string;
}

const EXPORT_OPTIONS: ExportOption[] = [
  // Two formats only.
  //
  // A design that leaves NEURAX is either being archived and re-imported, or
  // handed to the compiler — JSON covers the first, NEURAX IR the second. The
  // framework emitters that used to sit here (PyTorch, ONNX, Rust, Triton,
  // server config, network graph) produced skeletons that were never checked
  // against the model they claimed to represent, which is a worse promise than
  // not making one.
  { id: 'json', name: 'JSON', description: 'Architecture and analysis, re-importable', extension: '.json', icon: 'FileJson', minPlan: 'free' },
  { id: 'neurax-ir', name: 'NEURAX IR', description: 'Compiler input — the exact topology analysed', extension: '.neurax.json', icon: 'Box', includeAnalysis: true, minPlan: 'free' },
  { id: 'toml', name: 'TOML', description: 'Same topology, in a format meant to be hand-edited and diffed', extension: '.toml', icon: 'FileCode2', includeAnalysis: true, minPlan: 'free' },
  { id: 'github', name: 'GitHub', description: 'Push the architecture to a repository', extension: '', icon: 'Github', minPlan: 'free' },
];

interface ExportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  architectureName?: string;
  nodes?: CanvasNode[];
  connections?: Connection[];
  groups?: NodeGroup[];
  selectedArchitecture?: ArchitectureFamily;
  /** The analysis already on screen — used to cross-check that the code this
   * panel generates implies the same parameter count, not a different model
   * wearing the same name. Absent (e.g. nothing analysed yet) just skips
   * the check rather than blocking on it. */
  analysisResult?: AnalysisResult | null;
}

// Mock code previews


export function ExportPanel({
  isOpen,
  onClose,
  architectureName = 'architecture',
  nodes = [],
  connections = [],
  groups = [],
  selectedArchitecture = 'transformer',
  analysisResult = null,
}: ExportPanelProps) {
  const [selectedFormat, setSelectedFormat] = useState<string>('json');
  const [copied, setCopied] = useState(false);
  const [showGitHubExport, setShowGitHubExport] = useState(false);
  // Which "Push to GitHub" button was clicked — the topology-only default in
  // the Export Formats tab, or the verified project from the Full Project
  // tab. Determines whether GitHubExportPanel gets the generated code.
  const [githubExportMode, setGithubExportMode] = useState<'topology' | 'project'>('topology');
  const [showAssistant, setShowAssistant] = useState<string | null>(null);
  const [isZipping, setIsZipping] = useState(false);
  const { toast } = useToast();

  const { config: hwConfig } = useHardware();

  // Regenerated on every render from the current canvas + hyperparameters —
  // the same principle as the metrics already on screen: nothing here is
  // cached, so there is nothing that can go stale between an edit and an
  // export. Only computed when the panel actually needs it (nodes present).
  const project: ProjectExportResult | null = nodes.length > 0
    ? buildProjectFiles(nodes, connections, hwConfig, selectedArchitecture, architectureName, analysisResult)
    : null;

  // Compile NEURAX IR JSON from canvas graph
  const neuraxIR = compileToNeuraxIR(nodes, connections, {
    modelName: architectureName,
    family: selectedArchitecture,
    groups,
    ...hwConfig,
    // Training
    learningRate: hwConfig.learningRate,
    numEpochs: hwConfig.numEpochs,
    // Hardware
    gpuCount: hwConfig.gpuCount,
    gpuMemoryGb: hwConfig.gpuMemoryGb,
    // Data
    datasetSize: hwConfig.datasetSize,
    vocabSize: hwConfig.vocabSize,
    numClasses: hwConfig.numClasses,
  });
  const neuraxJson = JSON.stringify(neuraxIR, null, 2);

  // Same data as the JSON export, in TOML — see toToml's own doc comment.
  let neuraxToml: string;
  try {
    neuraxToml = toToml(neuraxIR);
  } catch (err) {
    neuraxToml = `# Could not render as TOML: ${err instanceof Error ? err.message : String(err)}\n`;
  }

  /**
   * Write the export out and tell the user where it went.
   *
   * On the desktop this is a system save dialog, so the description carries the
   * real path and a dismissed dialog produces no toast at all. In a browser the
   * download manager picks the location and only the filename is known.
   */
  async function saveAndNotify(
    content: string,
    filename: string,
    mimeType: string,
    title: string,
  ): Promise<boolean> {
    try {
      const result = await saveTextFile(content, filename, mimeType);
      if (!result.saved) return false;
      toast({ title, description: result.path ?? filename });
      return true;
    } catch (error) {
      toast({
        title: 'Export failed',
        description: error instanceof Error ? error.message : String(error),
        variant: 'destructive',
      });
      return false;
    }
  }

  const handleExport = async (format: ExportOption) => {
    if (nodes.length === 0) {
      toast({
        title: "No Architecture",
        description: "Add layers to the canvas before exporting",
        variant: "destructive",
      });
      return;
    }

    // ── NEURAX IR — the exact topology the compiler analyses ────────
    if (format.id === 'neurax-ir') {
      const saved = await saveAndNotify(
        neuraxJson,
        `${architectureName}.neurax.json`,
        'application/json',
        'NEURAX IR exported — the topology as analysed',
      );
      if (saved) onClose();
      return;
    }

    if (format.id === 'github') {
      setGithubExportMode('topology');
      setShowGitHubExport(true);
      return;
    }

    if (format.id === 'json') {
      const filename = `${architectureName.toLowerCase().replace(/\s+/g, '_')}.neurax.json`;
      const saved = await saveAndNotify(
        neuraxJson,
        filename,
        'application/json',
        'JSON export complete',
      );
      if (saved) onClose();
      return;
    }

    if (format.id === 'toml') {
      const filename = `${architectureName.toLowerCase().replace(/\s+/g, '_')}.toml`;
      const saved = await saveAndNotify(
        neuraxToml,
        filename,
        'application/toml',
        'TOML export complete',
      );
      if (saved) onClose();
      return;
    }

    // `EXPORT_OPTIONS` offers JSON, NEURAX IR and TOML and nothing else, so reaching
    // here means a format was added to the list without a branch to handle it.
    // The code that used to sit here handled `network`, `pytorch` and `rust`
    // — formats removed from the list — and ended in a fallback that showed
    // "Export Complete" a second later without having written anything.
    toast({
      title: 'Unsupported export format',
      description: `No handler for "${format.id}". Export as JSON, NEURAX IR, or TOML instead.`,
      variant: 'destructive',
    });
  };

  const handleDownloadProject = async () => {
    if (!project) return;
    setIsZipping(true);
    try {
      const result = await downloadProjectZip(project);
      if (result.saved) {
        toast({
          title: 'Project exported',
          description: result.path ?? `${project.slug}.zip`,
        });
      }
    } catch (error) {
      toast({
        title: 'Export failed',
        description: error instanceof Error ? error.message : String(error),
        variant: 'destructive',
      });
    } finally {
      setIsZipping(false);
    }
  };

  const handleCopyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast({
      title: "Copied",
      description: "Code copied to clipboard",
    });
  };

  const accessibleExports = EXPORT_OPTIONS;

  return (
    <>
      <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col bg-card">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Download className="w-5 h-5 text-primary" />
              Export Architecture
            </DialogTitle>
            <DialogDescription>
              Export your architecture in various formats with analysis data
            </DialogDescription>
          </DialogHeader>

          <Tabs defaultValue="formats" className="flex-1 overflow-hidden flex flex-col">
            <TabsList className="grid grid-cols-5 w-full">
              <TabsTrigger value="formats">Export Formats</TabsTrigger>
              <TabsTrigger value="neurax" className="flex items-center gap-1">
                <FileJson className="w-3 h-3 text-primary" />
                NEURAX IR
              </TabsTrigger>
              <TabsTrigger value="toml" className="flex items-center gap-1">
                <FileCode2 className="w-3 h-3 text-primary" />
                TOML
              </TabsTrigger>
              <TabsTrigger value="project" className="flex items-center gap-1">
                <Code className="w-3 h-3 text-primary" />
                Full Project
              </TabsTrigger>
            </TabsList>

            <TabsContent value="formats" className="flex-1 overflow-y-auto p-1">
              {/* Accessible formats */}
              <div className="grid grid-cols-2 gap-3 mt-2">
                {accessibleExports.map((format) => {
                  const Icon = iconMap[format.icon] || FileText;
                  const isSelected = selectedFormat === format.id;

                  return (
                    <button
                      key={format.id}
                      className={cn(
                        "p-4 rounded-lg border text-left transition-all",
                        isSelected
                          ? "bg-primary/10 border-primary"
                          : "bg-secondary/30 border-border hover:border-primary/50"
                      )}
                      onClick={() => setSelectedFormat(format.id)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <Icon className="w-5 h-5 text-primary" />
                        </div>
                        {isSelected && <Check className="w-5 h-5 text-primary" />}
                      </div>
                      <div className="text-sm font-medium mb-0.5">{format.name}</div>
                      <div className="text-[10px] text-muted-foreground mb-2">{format.description}</div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-[9px]">{format.extension}</Badge>
                        {format.includeAnalysis && (
                          <Badge className="text-[9px] bg-success/20 text-success border-0">
                            +Analysis
                          </Badge>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>



              <div className="mt-4 flex justify-end gap-2">
                <Button variant="outline" onClick={onClose}>
                  Cancel
                </Button>
                <Button
                  variant="outline"
                  onClick={() => { setGithubExportMode('topology'); setShowGitHubExport(true); }}
                >
                  <Github className="w-4 h-4 mr-2" />
                  Push to GitHub
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    const format = EXPORT_OPTIONS.find(f => f.id === selectedFormat);
                    if (format) setShowAssistant(format.name);
                  }}
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Export Assistant
                </Button>
                <Button
                  onClick={() => {
                    const format = EXPORT_OPTIONS.find(f => f.id === selectedFormat);
                    if (format) handleExport(format);
                  }}
                  disabled={false}
                >
                  {false ? (
                    <Loader2 key="loader" className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Download key="download" className="w-4 h-4 mr-2" />
                  )}
                  {false ? 'Exporting...' : `Export ${EXPORT_OPTIONS.find(f => f.id === selectedFormat)?.name}`}
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="neurax" className="flex-1 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <FileJson className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">NEURAX IR — Canonical JSON</span>
                  {nodes.length > 0 && (
                    <Badge variant="outline" className="text-[9px] bg-success/10 text-success border-success/30">
                      Compiled from canvas
                    </Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7"
                  onClick={() => handleCopyCode(neuraxJson)}
                >
                  {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                  {copied ? 'Copied!' : 'Copy'}
                </Button>
              </div>
              <div className="flex-1 overflow-auto bg-background rounded-lg border border-border">
                <pre className="p-4 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                  {neuraxJson}
                </pre>
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => handleCopyCode(neuraxJson)}>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy JSON
                </Button>
                <Button
                  size="sm"
                  onClick={() =>
                    saveAndNotify(
                      neuraxJson,
                      `${architectureName}.neurax.json`,
                      'application/json',
                      'NEURAX IR exported',
                    )
                  }
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download .neurax.json
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="toml" className="flex-1 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <FileCode2 className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">Same topology, as TOML</span>
                  {nodes.length > 0 && (
                    <Badge variant="outline" className="text-[9px] bg-success/10 text-success border-success/30">
                      Compiled from canvas
                    </Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7"
                  onClick={() => handleCopyCode(neuraxToml)}
                >
                  {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                  {copied ? 'Copied!' : 'Copy'}
                </Button>
              </div>
              <div className="flex-1 overflow-auto bg-background rounded-lg border border-border">
                <pre className="p-4 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                  {neuraxToml}
                </pre>
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => handleCopyCode(neuraxToml)}>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy TOML
                </Button>
                <Button
                  size="sm"
                  onClick={() =>
                    saveAndNotify(
                      neuraxToml,
                      `${architectureName}.toml`,
                      'application/toml',
                      'TOML exported',
                    )
                  }
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download .toml
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="project" className="flex-1 overflow-y-auto p-1 space-y-4">
              {!project ? (
                <div className="text-sm text-muted-foreground p-4">
                  Add layers to the canvas to generate a project.
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <FolderTree className="w-4 h-4 text-primary" />
                    <span className="text-sm font-medium">Runnable model + training project</span>
                    <Badge variant="outline" className="text-[9px] bg-success/10 text-success border-success/30">
                      Compiled from canvas
                    </Badge>
                  </div>

                  {project.codegen.fullySupported && !analysisResult ? (
                    <div className="rounded-lg border border-border bg-muted/40 p-3 flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div className="text-xs">
                        <div className="font-medium">
                          Not yet cross-checked — {project.codegen.totalParams.toLocaleString('en-US')} parameters generated
                        </div>
                        <div className="text-muted-foreground mt-0.5">
                          Run an analysis first so this export can be verified against it — an unverified
                          export is not the same guarantee as the numbers already shown elsewhere in NEURAX.
                        </div>
                      </div>
                    </div>
                  ) : project.codegen.fullySupported ? (
                    <div className="rounded-lg border border-success/30 bg-success/10 p-3 flex items-start gap-2">
                      <Check className="w-4 h-4 text-success mt-0.5 shrink-0" />
                      <div className="text-xs">
                        <div className="font-medium text-success">
                          Every layer translated — {project.codegen.totalParams.toLocaleString('en-US')} parameters generated
                        </div>
                        <div className="text-muted-foreground mt-0.5">
                          {project.verification.matches
                            ? `Matches NEURAX's analysis (within ${(project.verification.deltaPct * 100).toFixed(2)}%, ${project.attempts} attempt${project.attempts === 1 ? '' : 's'}). The code is a faithful reflection of the compiled design.`
                            : `Differs from NEURAX's analysis by ${(project.verification.deltaPct * 100).toFixed(1)}% — do not train from this export until that is resolved.`}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-lg border border-warning/30 bg-warning/10 p-3 flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-warning mt-0.5 shrink-0" />
                      <div className="text-xs">
                        <div className="font-medium text-warning">
                          {project.codegen.unsupportedTypes.length} block type(s) not translated
                        </div>
                        <div className="text-muted-foreground mt-0.5">
                          <code className="bg-muted px-1 rounded">{project.codegen.unsupportedTypes.join('`, `')}</code>{' '}
                          fall outside NEURAX's verified code-generation set (transformer / MoE / SSM / CNN-ResNet).
                          They raise <code className="bg-muted px-1 rounded">NotImplementedError</code> in the generated
                          file rather than silently doing nothing — fill them in by hand, or simplify the design.
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <Label className="text-sm font-medium">Files</Label>
                    <div className="rounded-md border border-border bg-muted/40 p-3 text-xs space-y-1 font-mono">
                      {project.files.map((f) => (
                        <div key={f.path}>{f.path}</div>
                      ))}
                    </div>
                  </div>

                  <div className="flex-1 overflow-auto bg-background rounded-lg border border-border max-h-64">
                    <pre className="p-4 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                      {project.codegen.code}
                    </pre>
                  </div>

                  <div className="flex justify-end gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleCopyCode(project.codegen.code)}
                    >
                      {copied ? <Check className="w-4 h-4 mr-2" /> : <Copy className="w-4 h-4 mr-2" />}
                      Copy model.py
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => { setGithubExportMode('project'); setShowGitHubExport(true); }}
                    >
                      <Github className="w-4 h-4 mr-2" />
                      Push to GitHub
                    </Button>
                    <Button size="sm" onClick={handleDownloadProject} disabled={isZipping}>
                      {isZipping ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      ) : (
                        <Download className="w-4 h-4 mr-2" />
                      )}
                      {isZipping ? 'Zipping...' : `Download ${project.slug}.zip`}
                    </Button>
                  </div>
                </>
              )}
            </TabsContent>

          </Tabs>

          {/* GitHub Export Panel */}
          <GitHubExportPanel
            isOpen={showGitHubExport}
            onClose={() => setShowGitHubExport(false)}
            nodes={nodes}
            connections={connections}
            modelName={architectureName}
            projectFiles={
              githubExportMode === 'project' && project?.codegen.fullySupported
                ? project.files
                : undefined
            }
          />
        </DialogContent>
      </Dialog>

      {/* Export Assistant Wizard — rendered outside Dialog portal */}
      <ExportAssistant
        isOpen={!!showAssistant}
        onClose={() => setShowAssistant(null)}
        format={showAssistant || 'PyTorch'}
        nodes={nodes}
        connections={connections}
        architectureName={architectureName}
        selectedArchitecture={selectedArchitecture}
      />
    </>
  );
}

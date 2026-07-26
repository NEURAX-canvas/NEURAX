import { useState, useCallback } from 'react';
import {
  Github,
  FolderGit2,
  GitBranch,
  FileCode,
  Check,
  X,
  Loader2,
  ExternalLink,
  Lock,
  Eye,
  Key,
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Checkbox } from '@/components/ui/checkbox.tsx';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog.tsx';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select.tsx';
import { useToast } from '@/hooks/use-toast.ts';

import { CanvasNode, Connection } from '@/types/architecture.ts';
import { generateCode, GeneratedCode } from '@/utils/codeGenerators.ts';
import { exportToGitHub, ExportGitHubFile } from '@/services/neuraxApi.ts';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';
import { cn } from '@/lib/utils.ts';

interface GitHubExportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  nodes: CanvasNode[];
  connections: Connection[];
  modelName?: string;
}

type ExportFormat = 'pytorch' | 'onnx' | 'rust' | 'triton' | 'json';

interface ExportFormatOption {
  id: ExportFormat;
  name: string;
  extension: string;
  description: string;
}

const EXPORT_FORMATS: ExportFormatOption[] = [
  { id: 'pytorch', name: 'PyTorch', extension: '.py', description: 'Python model definition' },
  { id: 'onnx', name: 'ONNX Export', extension: '.py', description: 'ONNX export script' },
  { id: 'json', name: 'JSON Schema', extension: '.json', description: 'Architecture schema' },
  { id: 'rust', name: 'Rust / Burn', extension: '.rs', description: 'Rust model structure' },
  { id: 'triton', name: 'Triton Kernels', extension: '.py', description: 'Optimized GPU kernels' },
];

export function GitHubExportPanel({
  isOpen,
  onClose,
  nodes,
  connections,
  modelName = 'GeneratedModel'
}: GitHubExportPanelProps) {
  const { toast } = useToast();
  // GitHub connection state
  const [isConnected, setIsConnected] = useState(false);
  const [githubToken, setGithubToken] = useState('');

  // Export configuration
  const [selectedRepo, setSelectedRepo] = useState('');
  const [branch, setBranch] = useState('main');
  const [directory, setDirectory] = useState('models/');
  const [commitMessage, setCommitMessage] = useState(`Add ${modelName} architecture from NEURAX`);
  const [createPR, setCreatePR] = useState(false);
  const [prBranch, setPrBranch] = useState(`neurax/${modelName.toLowerCase().replace(/\s+/g, '-')}`);

  // Selected formats
  const [selectedFormats, setSelectedFormats] = useState<ExportFormat[]>(['pytorch', 'json']);

  // Export state
  const [isExporting, setIsExporting] = useState(false);
  const [exportResult, setExportResult] = useState<{ success: boolean; url?: string; error?: string; fileUrls?: string[] } | null>(null);

  // Preview state
  const [previewCode, setPreviewCode] = useState<GeneratedCode | null>(null);

  // Repo list (user enters owner/repo manually)
  const repoPresets = [
    { id: 'user/model-hub', name: 'model-hub', fullName: 'user/model-hub' },
  ];

  const handleConnect = () => {
    const token = githubToken.trim();
    if (!token) {
      toast({
        title: 'Token Required',
        description: 'Enter a GitHub Personal Access Token with repo scope.',
        variant: 'destructive',
      });
      return;
    }
    // Validate token format (must be a GitHub PAT starting with ghp_ or github_pat_)
    if (!token.startsWith('ghp_') && !token.startsWith('github_pat_') && !token.startsWith('gho_')) {
      toast({
        title: 'Invalid Token Format',
        description: 'GitHub PATs start with "ghp_", "github_pat_", or "gho_".',
        variant: 'destructive',
      });
      return;
    }
    setIsConnected(true);
    toast({
      title: 'GitHub Connected',
      description: 'Token accepted. Select your repository and formats.',
    });
  };

  const handleDisconnect = () => {
    setIsConnected(false);
    setGithubToken('');
    setSelectedRepo('');
    setExportResult(null);
    toast({
      title: 'GitHub Disconnected',
      description: 'Your GitHub token has been cleared.',
    });
  };

  const toggleFormat = (format: ExportFormat) => {
    setSelectedFormats(prev =>
      prev.includes(format)
        ? prev.filter(f => f !== format)
        : [...prev, format]
    );
  };

  const handlePreview = (format: ExportFormat) => {
    const code = generateCode(format, nodes, connections, { modelName });
    setPreviewCode(code);
  };

  /** Build file list from selected export formats */
  const buildFiles = useCallback((): ExportGitHubFile[] => {
    const files: ExportGitHubFile[] = [];
    const dir = directory.replace(/\/+$/, ''); // strip trailing slash

    for (const fmt of selectedFormats) {
      const code = generateCode(fmt, nodes, connections, { modelName });
      files.push({
        path: `${dir}/${code.filename}`,
        content: code.content,
      });
    }

    // Always include the NEURAX IR JSON for reproducibility
    const ir = compileToNeuraxIR(nodes, connections);
    files.push({
      path: `${dir}/${modelName.toLowerCase().replace(/\s+/g, '_')}.neurax.json`,
      content: JSON.stringify(ir, null, 2),
    });

    return files;
  }, [selectedFormats, directory, nodes, connections, modelName]);

  const handleExport = async () => {
    if (!selectedRepo) {
      toast({
        title: 'Repository Required',
        description: 'Enter your GitHub repository as "owner/repo".',
        variant: 'destructive',
      });
      return;
    }

    if (selectedFormats.length === 0) {
      toast({
        title: 'No Formats Selected',
        description: 'Please select at least one export format.',
        variant: 'destructive',
      });
      return;
    }

    setIsExporting(true);
    setExportResult(null);

    try {
      const files = buildFiles();

      const result = await exportToGitHub({
        files,
        github_token: githubToken,
        repo: selectedRepo,
        branch: branch || 'main',
        commit_message: commitMessage,
        create_pr: createPR,
        pr_branch: createPR ? prBranch : undefined,
      });

      if (result.success) {
        setExportResult({
          success: true,
          url: result.pr_url || result.file_urls[0] || undefined,
          fileUrls: result.file_urls,
        });
        toast({
          title: 'Export Successful!',
          description: `${files.length} files pushed to ${selectedRepo}/${branch}`,
        });
      } else {
        setExportResult({
          success: false,
          error: result.error || 'Unknown error occurred',
        });
        toast({
          title: 'Export Failed',
          description: result.error || 'Unknown error',
          variant: 'destructive',
        });
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Network error';
      setExportResult({ success: false, error: message });
      toast({
        title: 'Export Failed',
        description: message,
        variant: 'destructive',
      });
    } finally {
      setIsExporting(false);
    }
  };

  const handleClose = () => {
    setExportResult(null);
    setPreviewCode(null);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col bg-card">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Github className="w-5 h-5" />
            Export to GitHub
          </DialogTitle>
          <DialogDescription>
            Push your generated code directly to your GitHub repository
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-6 py-4">
          {/* GitHub Token Input */}
          <div className="p-4 rounded-lg border border-border bg-secondary/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "w-10 h-10 rounded-lg flex items-center justify-center",
                  isConnected ? "bg-success/20" : "bg-muted"
                )}>
                  <Github className={cn(
                    "w-5 h-5",
                    isConnected ? "text-success" : "text-muted-foreground"
                  )} />
                </div>
                <div>
                  <div className="font-medium text-sm">
                    {isConnected ? 'GitHub Connected' : 'Connect GitHub'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {isConnected
                      ? 'Ready to push code to your repositories'
                      : 'Enter a Personal Access Token (classic or fine-grained)'
                    }
                  </div>
                </div>
              </div>

              {isConnected ? (
                <Button variant="outline" size="sm" onClick={handleDisconnect}>
                  Disconnect
                </Button>
              ) : (
                <span />
              )}
            </div>

            {!isConnected && (
              <div className="mt-3 flex gap-2">
                <div className="relative flex-1">
                  <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    type="password"
                    value={githubToken}
                    onChange={(e) => setGithubToken(e.target.value)}
                    className="pl-9 pr-9 font-mono text-xs"
                    placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    onClick={() => {
                      const input = document.querySelector<HTMLInputElement>('#github-token-input');
                      if (input) input.type = input.type === 'password' ? 'text' : 'password';
                    }}
                    title="Toggle visibility"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                </div>
                <Button size="sm" onClick={handleConnect}>
                  Connect
                </Button>
              </div>
            )}
          </div>

          {isConnected && (
            <>
              {/* Repository Input */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Repository</Label>
                <Select value={selectedRepo} onValueChange={setSelectedRepo}>
                  <SelectTrigger>
                    <SelectValue placeholder="Type owner/repo (e.g. user/my-model)" />
                  </SelectTrigger>
                  <SelectContent>
                    {repoPresets.map((repo) => (
                      <SelectItem key={repo.id} value={repo.fullName}>
                        <div className="flex items-center gap-2">
                          <FolderGit2 className="w-4 h-4" />
                          <span className="truncate">{repo.fullName}</span>
                          <Lock className="w-3 h-3 text-muted-foreground" />
                        </div>
                      </SelectItem>
                    ))}
                    {/* Custom repo input */}
                    <div className="p-2 border-t border-border">
                      <div className="flex gap-2">
                        <Input
                          placeholder="owner/repo-name"
                          value={repoPresets.includes({ id: selectedRepo } as any) ? '' : selectedRepo}
                          onChange={(e) => setSelectedRepo(e.target.value)}
                          className="text-xs h-8"
                        />
                      </div>
                    </div>
                  </SelectContent>
                </Select>
                <div className="text-[10px] text-muted-foreground">
                  Enter the full repository name, e.g. <code className="bg-muted px-1 rounded">my-org/my-model</code>
                </div>
              </div>

              {/* Export Formats Selection */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Export Formats</Label>
                <div className="grid grid-cols-2 gap-2">
                  {EXPORT_FORMATS.map((format) => {
                    const isSelected = selectedFormats.includes(format.id);

                    return (
                      <div
                        key={format.id}
                        className={cn(
                          "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all",
                          isSelected
                            ? "border-primary bg-primary/5"
                            : "border-border hover:border-primary/50"
                        )}
                        onClick={() => toggleFormat(format.id)}
                      >
                        <Checkbox
                          checked={isSelected}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">{format.name}</span>
                            <Badge variant="outline" className="text-[9px]">
                              {format.extension}
                            </Badge>
                          </div>
                          <div className="text-[10px] text-muted-foreground truncate">
                            {format.description}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 text-[10px]"
                          onClick={(e) => {
                            e.stopPropagation();
                            handlePreview(format.id);
                          }}
                        >
                          Preview
                        </Button>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Branch & Directory */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm">Branch</Label>
                  <div className="relative">
                    <GitBranch className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      value={branch}
                      onChange={(e) => setBranch(e.target.value)}
                      className="pl-9"
                      placeholder="main"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm">Directory</Label>
                  <div className="relative">
                    <FileCode className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      value={directory}
                      onChange={(e) => setDirectory(e.target.value)}
                      className="pl-9"
                      placeholder="models/"
                    />
                  </div>
                </div>
              </div>

              {/* Commit Message */}
              <div className="space-y-2">
                <Label className="text-sm">Commit Message</Label>
                <Input
                  value={commitMessage}
                  onChange={(e) => setCommitMessage(e.target.value)}
                  placeholder="Add model architecture"
                />
              </div>

              {/* Create PR Option */}
              <div className="flex items-center justify-between p-3 rounded-lg border border-border">
                <div className="space-y-0.5">
                  <Label className="text-sm font-medium">Create Pull Request</Label>
                  <p className="text-[10px] text-muted-foreground">
                    Create a new branch and open a PR instead of pushing directly
                  </p>
                </div>
                <Switch checked={createPR} onCheckedChange={setCreatePR} />
              </div>

              {createPR && (
                <div className="space-y-2 pl-4 border-l-2 border-primary/30">
                  <Label className="text-sm">PR Branch Name</Label>
                  <Input
                    value={prBranch}
                    onChange={(e) => setPrBranch(e.target.value)}
                    placeholder="neurax/model-name"
                  />
                </div>
              )}

              {/* Code Preview */}
              {previewCode && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">
                      Preview: {previewCode.filename}
                    </Label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setPreviewCode(null)}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="max-h-48 overflow-auto bg-background rounded-lg border border-border">
                    <pre className="p-3 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                      {previewCode.content}
                    </pre>
                  </div>
                </div>
              )}

              {/* Export Result */}
              {exportResult && (
                <div className={cn(
                  "p-4 rounded-lg border",
                  exportResult.success
                    ? "bg-success/10 border-success/30"
                    : "bg-destructive/10 border-destructive/30"
                )}>
                  <div className="flex items-center gap-2">
                    {exportResult.success ? (
                      <>
                        <Check className="w-5 h-5 text-success" />
                        <div className="flex-1">
                          <div className="font-medium text-sm text-success">Export Successful!</div>
                          <div className="text-xs text-muted-foreground">
                            Your code has been pushed to GitHub
                          </div>
                          {exportResult.fileUrls && exportResult.fileUrls.length > 0 && (
                            <div className="mt-2 space-y-0.5">
                              {exportResult.fileUrls.map((url, i) => (
                                <a
                                  key={i}
                                  href={url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs text-primary hover:underline flex items-center gap-1"
                                >
                                  <ExternalLink className="w-3 h-3" />
                                  {url.split('/').pop()}
                                </a>
                              ))}
                            </div>
                          )}
                        </div>
                        {exportResult.url && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => window.open(exportResult.url, '_blank')}
                          >
                            <ExternalLink className="w-4 h-4 mr-1" />
                            {exportResult.fileUrls && exportResult.fileUrls.length > 1 ? 'PR' : 'View'}
                          </Button>
                        )}
                      </>
                    ) : (
                      <>
                        <X className="w-5 h-5 text-destructive" />
                        <div className="flex-1">
                          <div className="font-medium text-sm text-destructive">Export Failed</div>
                          <div className="text-xs text-muted-foreground">
                            {exportResult.error || 'Please check your connection and try again'}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            onClick={handleExport}
            disabled={!isConnected || !selectedRepo || selectedFormats.length === 0 || isExporting}
          >
            {isExporting ? (
              <>
                <Loader2 key="loader" className="w-4 h-4 mr-2 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Github key="github" className="w-4 h-4 mr-2" />
                Push to GitHub
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

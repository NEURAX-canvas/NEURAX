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
import { Switch } from '@/components/ui/switch.tsx';
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
import { exportToGitHub, ExportGitHubFile } from '@/services/neuraxApi.ts';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';
import { cn } from '@/lib/utils.ts';

interface GitHubExportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  nodes: CanvasNode[];
  connections: Connection[];
  modelName?: string;
  /**
   * A verified, runnable project (`src/model.py`, `train.py`, ...) to push
   * instead of the topology-only default — from `ExportPanel`'s "Full
   * Project" tab, already generated and cross-checked against the analysis
   * before this panel ever sees it. `buildFiles` still refuses to invent
   * anything on its own; this is the one path that has already earned the
   * exception, by construction, not by trust.
   */
  projectFiles?: ExportGitHubFile[];
}


/** A README describing exactly what was pushed, and nothing more. */
function buildReadme(modelName: string, blocks: number, links: number): string {
  return [
    `# ${modelName}`,
    '',
    'Architecture designed with [NEURAX](https://github.com/rustnew/NEURAX),',
    'an analytical compiler for neural network architectures.',
    '',
    `- Blocks: ${blocks}`,
    `- Connections: ${links}`,
    '',
    '## What is in this directory',
    '',
    '`*.neurax.json` is the topology as the compiler analyses it. Re-open it in',
    'the NEURAX studio, or analyse it from the command line:',
    '',
    '```bash',
    'cargo install neurax-cli',
    `neurax analyze ${modelName.toLowerCase().replace(/\s+/g, '_')}.neurax.json`,
    '```',
    '',
    'This is a description of an architecture, not a trained model and not an',
    'implementation.',
    '',
  ].join('\n');
}

export function GitHubExportPanel({
  isOpen,
  onClose,
  nodes,
  connections,
  modelName = 'GeneratedModel',
  projectFiles,
}: GitHubExportPanelProps) {
  const { toast } = useToast();
  // GitHub connection state
  const [isConnected, setIsConnected] = useState(false);
  const [githubToken, setGithubToken] = useState('');

  // Export configuration
  const [selectedRepo, setSelectedRepo] = useState('');
  // Left blank by default rather than hardcoded to "main": an unset branch
  // tells the backend to use whatever the repository's actual default
  // branch is (creating the repository first if it doesn't exist yet) —
  // typing one here overrides that.
  const [branch, setBranch] = useState('');
  const [directory, setDirectory] = useState('models/');
  const [commitMessage, setCommitMessage] = useState(`Add ${modelName} architecture from NEURAX`);
  const [createPR, setCreatePR] = useState(false);
  const [prBranch, setPrBranch] = useState(`neurax/${modelName.toLowerCase().replace(/\s+/g, '-')}`);
  const [makePrivate, setMakePrivate] = useState(true);

  // Selected formats

  // Export state
  const [isExporting, setIsExporting] = useState(false);
  const [exportResult, setExportResult] = useState<{ success: boolean; url?: string; error?: string; fileUrls?: string[] } | null>(null);


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


  /**
   * What gets pushed.
   *
   * The architecture, and nothing invented. This used to also push generated
   * PyTorch and Rust — a class whose `__init__` was empty and whose `forward`
   * was a chain of `x2 = x1`, committed to the user's repository under the
   * name of their model. A file that claims to be LLaMA and is an identity
   * function is worse in someone's repository than absent from it.
   */
  const buildFiles = useCallback((): ExportGitHubFile[] => {
    const dir = directory.replace(/\/+$/, ''); // strip trailing slash
    const slug = modelName.toLowerCase().replace(/\s+/g, '_');

    if (projectFiles && projectFiles.length > 0) {
      return projectFiles.map((f) => ({ ...f, path: `${dir}/${f.path}` }));
    }

    const ir = compileToNeuraxIR(nodes, connections);
    return [
      {
        path: `${dir}/${slug}.neurax.json`,
        content: JSON.stringify(ir, null, 2),
      },
      {
        path: `${dir}/README.md`,
        content: buildReadme(modelName, nodes.length, connections.length),
      },
    ];
  }, [directory, nodes, connections, modelName, projectFiles]);

  const handleExport = async () => {
    if (!selectedRepo) {
      toast({
        title: 'Repository Required',
        description: 'Enter your GitHub repository as "owner/repo".',
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
        branch: branch.trim() || undefined,
        private: makePrivate,
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
                          value={repoPresets.some((repo) => repo.fullName === selectedRepo) ? '' : selectedRepo}
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

              {/* What gets pushed. Either the topology + README — the only
                  thing NEURAX can put in a repository truthfully by default
                  — or, when `projectFiles` was handed in already generated
                  and verified, the real runnable project. Never anything
                  invented in between. */}
              <div className="space-y-2">
                <Label className="text-sm font-medium">Files</Label>
                <div className="rounded-md border border-border bg-muted/40 p-3 text-xs space-y-1">
                  {(projectFiles && projectFiles.length > 0 ? projectFiles : buildFiles()).map((f) => (
                    <div key={f.path} className="font-mono">
                      {directory.replace(/\/+$/, '')}/{f.path.replace(/^\/+/, '')}
                    </div>
                  ))}
                  <p className="text-muted-foreground pt-1">
                    {projectFiles && projectFiles.length > 0
                      ? 'A verified, runnable project — the generated code was already cross-checked against the analysis before reaching this dialog.'
                      : 'The topology exactly as the compiler analyses it, re-openable in the studio and analysable with the CLI.'}
                  </p>
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
                      placeholder="auto-detected"
                    />
                  </div>
                  <p className="text-[10px] text-muted-foreground">
                    Leave blank to use the repository's actual default branch — NEURAX creates
                    the repository if it doesn't exist yet, so there's nothing to set up first.
                  </p>
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

              {/* Private-by-default. Only matters if NEURAX ends up creating the
                  repository — an existing one keeps whatever visibility it
                  already has, this cannot change that. */}
              <div className="flex items-center justify-between p-3 rounded-lg border border-border">
                <div className="space-y-0.5">
                  <Label className="text-sm font-medium flex items-center gap-1.5">
                    <Lock className="w-3.5 h-3.5" />
                    Private, if NEURAX creates the repository
                  </Label>
                  <p className="text-[10px] text-muted-foreground">
                    Only applies when the repository above doesn't exist yet — an existing
                    repository's visibility is never changed
                  </p>
                </div>
                <Switch checked={makePrivate} onCheckedChange={setMakePrivate} />
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
            disabled={!isConnected || !selectedRepo || isExporting}
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

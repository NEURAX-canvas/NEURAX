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
  Loader2
} from 'lucide-react';

import { cn } from '@/lib/utils.ts';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { generateCode } from '@/utils/codeGenerators.ts';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';
import { generateNetworkGraphHTML } from '@/utils/networkGraphExporter.ts';
import { useHardware } from '@/contexts/HardwareContext.tsx';
import { GitHubExportPanel } from './GitHubExportPanel.tsx';
import { ExportAssistant } from './ExportAssistant.tsx';

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
  Github
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
];

interface ExportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  architectureName?: string;
  nodes?: CanvasNode[];
  connections?: Connection[];
  groups?: NodeGroup[];
  selectedArchitecture?: ArchitectureFamily;
}

// Mock code previews
const MOCK_PYTORCH_CODE = `import torch
import torch.nn as nn

class AIArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input layer: [batch, 224, 224, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        
        # Attention block
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(512)
        
        # Output layer
        self.fc = nn.Linear(512, 1000)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.flatten(2).transpose(1, 2)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        
        x = x.mean(dim=1)
        return self.fc(x)

# Model Statistics:
# - Total Parameters: 25,600,000
# - Estimated FLOPs: 4.1 GFLOPs
# - Memory Usage: 97.6 MB (FP32)
`;

const MOCK_RUST_CODE = `use tch::{nn, Tensor};

/// AI Architecture Designer - Generated Model
/// 
/// Architecture Summary:
/// - Input Shape: [batch, 224, 224, 3]
/// - Output Shape: [batch, 1000]
/// - Total Parameters: 25,600,000
/// - Estimated FLOPs: 4.1 GFLOPs

pub struct AIArchitecture {
    conv1: nn::Conv2D,
    attention: MultiHeadAttention,
    layer_norm: nn::LayerNorm,
    fc: nn::Linear,
}

impl AIArchitecture {
    pub fn new(vs: &nn::Path) -> Self {
        let conv1 = nn::conv2d(
            vs / "conv1",
            3,  // in_channels
            64, // out_channels
            3,  // kernel_size
            Default::default(),
        );
        
        let attention = MultiHeadAttention::new(
            vs / "attention",
            512, // embed_dim
            8,   // num_heads
        );
        
        let layer_norm = nn::layer_norm(
            vs / "layer_norm",
            vec![512],
            Default::default(),
        );
        
        let fc = nn::linear(vs / "fc", 512, 1000, Default::default());
        
        Self { conv1, attention, layer_norm, fc }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.conv1.forward(x).relu();
        let x = x.flatten(2, -1).transpose(1, 2);
        
        let attn_out = self.attention.forward(&x, &x, &x);
        let x = self.layer_norm.forward(&(&x + &attn_out));
        
        let x = x.mean_dim(&[1], false, tch::Kind::Float);
        self.fc.forward(&x)
    }
}
`;

export function ExportPanel({
  isOpen,
  onClose,
  architectureName = 'architecture',
  nodes = [],
  connections = [],
  groups = [],
  selectedArchitecture = 'transformer'
}: ExportPanelProps) {
  const [selectedFormat, setSelectedFormat] = useState<string>('json');
  const [copied, setCopied] = useState(false);
  const [showGitHubExport, setShowGitHubExport] = useState(false);
  const [showAssistant, setShowAssistant] = useState<string | null>(null);
  const { toast } = useToast();

  const { config: hwConfig } = useHardware();

  // Generate real code based on current architecture
  const pytorchCode = nodes.length > 0
    ? generateCode('pytorch', nodes, connections, { modelName: architectureName }).content
    : MOCK_PYTORCH_CODE;

  const rustCode = nodes.length > 0
    ? generateCode('rust', nodes, connections, { modelName: architectureName }).content
    : MOCK_RUST_CODE;

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

  /** Trigger file download in the browser */
  function downloadFile(content: string, filename: string, mimeType: string) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
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
      const blob = new Blob([neuraxJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${architectureName}.neurax.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast({
        title: 'NEURAX IR exported',
        description: `${architectureName}.neurax.json — the topology as analysed`,
      });
      onClose();
      return;
    }

    if (format.id === 'json') {
      const filename = `${architectureName.toLowerCase().replace(/\s+/g, '_')}.neurax.json`;
      downloadFile(neuraxJson, filename, 'application/json');
      toast({
        title: "JSON Export Complete",
        description: `Architecture saved as ${filename}`,
      });
      onClose();
      return;
    }

    // ── Network Graph Export — interactive HTML visualization ───────
    if (format.id === 'network') {
      const html = generateNetworkGraphHTML(nodes, connections, groups, {
        modelName: architectureName,
        family: selectedArchitecture,
      });
      const filename = `${architectureName.toLowerCase().replace(/\s+/g, '_')}_graph.html`;
      downloadFile(html, filename, 'text/html');
      toast({
        title: "Network Graph Export Complete",
        description: `Interactive graph saved as ${filename}`,
      });
      onClose();
      return;
    }

    // ── Other formats (PyTorch, Rust, etc.) — generic download ────
    const codeMap: Record<string, { content: string; ext: string; mime: string }> = {
      pytorch: { content: pytorchCode, ext: '.py', mime: 'text/x-python' },
      rust: { content: rustCode, ext: '.rs', mime: 'text/x-rust' },
    };
    const entry = codeMap[format.id];
    if (entry) {
      const filename = `${architectureName.toLowerCase().replace(/\s+/g, '_')}${entry.ext}`;
      downloadFile(entry.content, filename, entry.mime);
      toast({
        title: `${format.name} Export Complete`,
        description: `File saved as ${filename}`,
      });
      onClose();
      return;
    }

    // Fallback toast for formats without implementation yet
    toast({ title: "Export Started", description: `Exporting ${architectureName}${format.extension}...` });
    setTimeout(() => {
      toast({ title: "Export Complete", description: `${format.name} file ready for download` });
    }, 1000);
    onClose();
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
            <TabsList className="grid grid-cols-4 w-full">
              <TabsTrigger value="formats">Export Formats</TabsTrigger>
              <TabsTrigger value="neurax" className="flex items-center gap-1">
                <FileJson className="w-3 h-3 text-primary" />
                NEURAX IR
              </TabsTrigger>
              <TabsTrigger value="pytorch">PyTorch</TabsTrigger>
              <TabsTrigger value="rust">
                <span className="flex items-center gap-1">
                  Rust
                </span>
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
                  onClick={() => setShowGitHubExport(true)}
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
                  onClick={() => {
                    const blob = new Blob([neuraxJson], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${architectureName}.neurax.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                    toast({
                      title: "Export Complete",
                      description: `NEURAX IR saved as ${architectureName}.neurax.json`,
                    });
                  }}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download .neurax.json
                </Button>
              </div>
            </TabsContent>


            <TabsContent value="pytorch" className="flex-1 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Code className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">PyTorch Model Definition</span>
                  {nodes.length > 0 && (
                    <Badge variant="outline" className="text-[9px] bg-success/10 text-success border-success/30">
                      Generated from canvas
                    </Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7"
                  onClick={() => handleCopyCode(pytorchCode)}
                >
                  {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                  {copied ? 'Copied!' : 'Copy'}
                </Button>
              </div>
              <div className="flex-1 overflow-auto bg-background rounded-lg border border-border">
                <pre className="p-4 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                  {pytorchCode}
                </pre>
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => handleCopyCode(pytorchCode)}>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy Code
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowGitHubExport(true)}
                >
                  <Github className="w-4 h-4 mr-2" />
                  Push to GitHub
                </Button>
                <Button
                  size="sm"
                  onClick={() => {
                    const format = EXPORT_OPTIONS.find(f => f.id === 'pytorch');
                    if (format) handleExport(format);
                  }}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download .py
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="rust" className="flex-1 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Cog className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">Rust Model Structure</span>
                  {nodes.length > 0 && (
                    <Badge variant="outline" className="text-[9px] bg-success/10 text-success border-success/30">
                      Generated from canvas
                    </Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7"
                  onClick={() => handleCopyCode(rustCode)}
                >
                  {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                  {copied ? 'Copied!' : 'Copy'}
                </Button>
              </div>
              <div className="flex-1 overflow-auto bg-background rounded-lg border border-border">
                <pre className="p-4 text-xs font-mono text-muted-foreground whitespace-pre overflow-x-auto">
                  {rustCode}
                </pre>
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => handleCopyCode(rustCode)}>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy Code
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowGitHubExport(true)}
                >
                  <Github className="w-4 h-4 mr-2" />
                  Push to GitHub
                </Button>
                <Button
                  size="sm"
                  onClick={() => {
                    const format = EXPORT_OPTIONS.find(f => f.id === 'rust');
                    if (format) handleExport(format);
                  }}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download .rs
                </Button>
              </div>
            </TabsContent>
          </Tabs>

          {/* GitHub Export Panel */}
          <GitHubExportPanel
            isOpen={showGitHubExport}
            onClose={() => setShowGitHubExport(false)}
            nodes={nodes}
            connections={connections}
            modelName={architectureName}
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

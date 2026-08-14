/**
 * The user guide, as data.
 *
 * Kept as structured content rather than a folder of Markdown files, for two
 * reasons. The desktop application ships as a single bundle with no network
 * and no filesystem reads at runtime, so documentation that lives in files
 * would have to be fetched or inlined at build time anyway. And keeping it
 * typed means the search index, the table of contents and the panel all read
 * the same source — documentation cannot fall out of step with its own
 * navigation.
 *
 * What matters more than the format: **everything stated here is true of this
 * build**. A guide that describes a button that is not there costs more than no
 * guide at all, because it teaches the reader to distrust the rest. Where
 * NEURAX is approximate or incomplete, the guide says so in the same voice it
 * uses for everything else — see the "Accuracy and limits" section, which is
 * not an appendix but part of knowing how to use the tool.
 */

/** A block of content inside a section. */
export type DocBlock =
  | { kind: 'text'; text: string }
  | { kind: 'heading'; text: string }
  | { kind: 'list'; items: string[] }
  | { kind: 'steps'; items: string[] }
  | { kind: 'code'; text: string; caption?: string }
  | { kind: 'keys'; items: Array<{ keys: string; action: string }> }
  | { kind: 'table'; columns: string[]; rows: string[][] }
  | { kind: 'note'; tone: 'info' | 'warning'; title: string; text: string };

export interface DocSection {
  id: string;
  title: string;
  /** One line shown under the title in the contents. */
  summary: string;
  blocks: DocBlock[];
}

export interface DocChapter {
  id: string;
  title: string;
  sections: DocSection[];
}

export const DOCUMENTATION: DocChapter[] = [
  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'start',
    title: 'Getting started',
    sections: [
      {
        id: 'what-is-neurax',
        title: 'What NEURAX is',
        summary: 'An analytical compiler for neural architectures — and what that means.',
        blocks: [
          {
            kind: 'text',
            text:
              'NEURAX answers questions about a neural network **before you train it**. You describe an architecture — by drawing it, loading a template, or importing a config — and NEURAX computes what it would cost to build: how much VRAM it needs, how many FLOPs it takes, how long training would run, what that costs in dollars and in carbon.',
          },
          {
            kind: 'text',
            text:
              'It does this analytically. {+There is no GPU involved+} and nothing is executed. The answers come from formulas over the architecture, which is why they arrive in {+well under a second+} and why {+the same design always produces the same numbers+}.',
          },
          { kind: 'heading', text: 'What it is not' },
          {
            kind: 'list',
            items: [
              '**Not a training framework.** {-NEURAX never runs your model.-} PyTorch and JAX do that.',
              '**Not a runtime compiler.** IREE and OpenXLA lower a model for execution; NEURAX works at design time, before there is anything to execute.',
              '**Not a profiler.** A profiler measures a model that ran. NEURAX predicts one that has not.',
            ],
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'The question NEURAX is built for',
            text:
              '"I am considering 32 layers at width 4096 with grouped-query attention. Will it fit on eight H100s, and what will a full training run cost me?" That question has a real answer before any GPU is booked, and finding it should take seconds rather than an afternoon.',
          },
        ],
      },
      {
        id: 'first-analysis',
        title: 'Your first analysis',
        summary: 'From an empty canvas to a full report in about a minute.',
        blocks: [
          {
            kind: 'text',
            text:
              'The fastest way to see what NEURAX does is to load a model that already exists and analyse it.',
          },
          {
            kind: 'steps',
            items: [
              'Open **Templates** in the toolbar and pick a model — LLaMA 2 7B is a good first choice.',
              'The canvas fills with the architecture. Each box is a block; the lines are the tensors flowing between them.',
              'Press **Run Analysis** at the top right.',
              'The right-hand panel fills with results: parameter count, peak VRAM, FLOPs, latency, training cost.',
              'Click any block on the canvas. The inspector at the bottom shows what that block contributes on its own.',
            ],
          },
          {
            kind: 'text',
            text:
              'Now change something. Open **Hyperparameters**, raise the layer count, and run the analysis again. Every figure moves. That loop — change, analyse, read — is the whole tool.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Nothing leaves your machine',
            text:
              'In the desktop application the compiler runs inside the process, on a loopback socket. {+No architecture you design is sent anywhere+}, and {+NEURAX works with no internet connection+}. The single exception is the AI Copilot, which calls the provider whose API key you supply.',
          },
        ],
      },
      {
        id: 'workspace-tour',
        title: 'The workspace',
        summary: 'What each part of the window is for.',
        blocks: [
          {
            kind: 'table',
            columns: ['Area', 'What it does'],
            rows: [
              ['**Block palette** (left)', 'Every block available for the current architecture family. Drag one onto the canvas to add it.'],
              ['**Canvas** (centre)', 'The architecture itself. Drag to move blocks, drag from a port to connect them, scroll to zoom.'],
              ['**Inspector** (bottom)', "The selected block's parameters, and what it contributes to the totals."],
              ['**Analysis panel** (right)', 'The results, in three tabs: Architecture, Performance, Hardware.'],
              ['**Workspace tabs** (top)', 'Five views of the same design — see below.'],
            ],
          },
          { kind: 'heading', text: 'The five workspaces' },
          {
            kind: 'table',
            columns: ['Workspace', 'Purpose'],
            rows: [
              ['**Architecture**', 'Design the model. This is where you spend most of your time.'],
              ['**Simulation**', 'Training-time behaviour: memory over a step, throughput, per-layer breakdowns.'],
              ['**Production**', 'What deploying this model looks like.'],
              ['**Inference Intelligence**', 'Serving-time behaviour — stability, entropy, attention focus, hallucination risk — across 22 sampling and context settings.'],
              ['**Time Machine**', 'Multi-year projections of cost, carbon and scaling, with regulatory tracking.'],
            ],
          },
          {
            kind: 'text',
            text:
              'All five read the same design. Switching tabs never changes your architecture.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'designing',
    title: 'Designing a model',
    sections: [
      {
        id: 'canvas-basics',
        title: 'Working on the canvas',
        summary: 'Adding, connecting, grouping and deleting blocks.',
        blocks: [
          { kind: 'heading', text: 'Blocks' },
          {
            kind: 'list',
            items: [
              'Drag a block from the palette onto the canvas to add it.',
              'Click a block to select it; its parameters appear in the inspector.',
              'Drag a block to move it. Position is cosmetic — it does not affect the analysis.',
              'Select several blocks with a rubber-band drag, or hold **Shift** while clicking.',
            ],
          },
          { kind: 'heading', text: 'Connections' },
          {
            kind: 'text',
            text:
              'Drag from the output port on the right of one block to the input port on the left of another. NEURAX validates the connection as you make it and refuses ones that cannot carry a tensor. Click a connection to select it, then press **Delete** to remove it.',
          },
          { kind: 'heading', text: 'Groups and repeated blocks' },
          {
            kind: 'text',
            text:
              'Select two or more blocks and press **Ctrl+G** to group them. A group can carry a repeat count, which is how a transformer describes "32 identical decoder blocks" without drawing thirty-two copies. The `layer_stack` block does the same thing for a whole layer stack: the blocks connected to it are the body of one layer, and its `num_layers` says how many times that body repeats.',
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'The layer stack is load-bearing',
            text:
              'If the blocks forming a layer body are not connected back to their `layer_stack`, the compiler counts them once rather than once per layer — and {-every memory, cost and FLOP figure downstream is wrong by that factor-}. When a template looks right but the parameter count is far too low, this is almost always why.',
          },
        ],
      },
      {
        id: 'families',
        title: 'Architecture families',
        summary: 'Eleven families, and why the choice matters.',
        blocks: [
          {
            kind: 'text',
            text:
              'The family selector at the top of the window decides two things: which blocks the palette offers, and which formulas the compiler applies. A mixture-of-experts model analysed as a plain transformer would have its experts counted once instead of per expert.',
          },
          {
            kind: 'list',
            items: [
              '**Transformer** — encoder, decoder and encoder-decoder models.',
              '**CNN** — convolutional vision models.',
              '**MoE** — mixture of experts, with routing and expert blocks.',
              '**SSM** — state-space models such as Mamba and RWKV.',
              '**Diffusion** — denoising models, U-Nets and DiTs.',
              '**GNN** — graph networks.',
              '**GAN** — generator and discriminator pairs.',
              '**RL** — policy and value networks.',
              '**SNN** — spiking networks.',
              '**RNN** — LSTM and GRU models.',
              '**Experimental** — blocks that do not belong to an established family.',
            ],
          },
          {
            kind: 'text',
            text:
              'The catalogue holds **208 configurable blocks** across these families, and **88 reference templates** built from them.',
          },
        ],
      },
      {
        id: 'hyperparameters',
        title: 'Hyperparameters and hardware',
        summary: 'The settings every metric is computed against.',
        blocks: [
          {
            kind: 'text',
            text:
              'Two dialogs in the toolbar set the terms of the analysis. They matter as much as the architecture — the same model on different hardware, or at a different precision, gives entirely different numbers.',
          },
          { kind: 'heading', text: 'Hyperparameters' },
          {
            kind: 'text',
            text:
              'Model dimensions (width, depth, heads, vocabulary, sequence length), training settings (batch size, learning rate, epochs, optimizer, gradient checkpointing), and parallelism (tensor, pipeline, expert, ZeRO stage). Family-specific settings appear for the family you have selected — expert count and top-k for MoE, state dimension for SSM, and so on.',
          },
          { kind: 'heading', text: 'Target' },
          {
            kind: 'text',
            text:
              'The chip every metric is computed for: which GPU, how many, and how they are connected. NEURAX ships specifications for more than twenty GPUs — A100, H100, V100, the RTX 40 series and others — with their memory, bandwidth and per-precision throughput.',
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'Precision changes every memory figure',
            text:
              'Switching between fp32, bf16 and fp8 changes weights, activations, gradients and optimizer state at once. When comparing two designs, {-keep the precision the same-} or you are measuring the precision, not the architecture.',
          },
        ],
      },
      {
        id: 'copilot',
        title: 'The AI Copilot',
        summary: 'Describing an architecture in words, with your own API key.',
        blocks: [
          {
            kind: 'text',
            text:
              'The Copilot builds and edits architectures from natural language — "create a transformer for image classification", "add grouped-query attention", "why is this running out of memory". It can also validate a topology and suggest changes.',
          },
          {
            kind: 'text',
            text:
              'It works on **your own key**, for OpenAI, Anthropic, Google or Mistral. The key is stored in your browser or desktop profile and is sent only to the provider you chose — {+never to NEURAX+}.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Check what it builds',
            text:
              'The Copilot is a fast way to get a first draft onto the canvas. {-It is not a source of truth about your model-}: read the blocks it produced and run the analysis before trusting the shape.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'importing',
    title: 'Bringing models in',
    sections: [
      {
        id: 'import-hf',
        title: 'Importing a HuggingFace config.json',
        summary: 'Load any open-weights model from the file that defines its shape.',
        blocks: [
          {
            kind: 'text',
            text:
              'Almost every model on the HuggingFace Hub ships a `config.json` beside its weights, and that file fully determines the architecture. NEURAX reads it directly, so working on a model that already exists does not mean rebuilding it block by block.',
          },
          {
            kind: 'steps',
            items: [
              'Press **Import** in the toolbar.',
              'Either **Open File** and pick a `config.json`, or paste its contents into the box.',
              'NEURAX detects the format itself — the badge above the box says what it recognised.',
              'Press **Read File**. A summary appears: layer count, width, heads, vocabulary, and the structural choices it inferred.',
              'Press **Import**. The architecture appears on the canvas and the analysis runs.',
            ],
          },
          { kind: 'heading', text: 'What it reads' },
          {
            kind: 'text',
            text:
              'Widths, depth, attention and key-value head counts, head dimension, feed-forward width, vocabulary, context length, normalisation epsilon, RoPE base, activation, weight tying, and expert counts for routed models. Both naming conventions are handled: `hidden_size` and `n_embd`, `num_hidden_layers` and `n_layer`, and so on.',
          },
          {
            kind: 'list',
            items: [
              '**Dense decoders** — LLaMA, Mistral, Qwen, Gemma, Phi, OLMo, StableLM, Falcon, GPT-2, GPT-NeoX, OPT, Bloom.',
              '**Encoders** — BERT, RoBERTa, DistilBERT, ELECTRA and relatives, which receive a classification head rather than an LM head.',
              '**Mixture of experts** — Mixtral, Qwen-MoE and DeepSeek-style configs, including shared experts.',
              '**Multimodal** — the language tower is read from `text_config`; vision and audio towers are not imported.',
            ],
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'It tells you what it assumed',
            text:
              'Configs vary in what they state. Where a field is missing, NEURAX picks the conventional default and lists every one of those choices in an amber box before you import — so {+a guess is never silent+}.',
          },
          { kind: 'heading', text: 'How accurate is it?' },
          {
            kind: 'text',
            text:
              'Imported dense models are checked against their published parameter counts, end to end through the real compiler, on every build:',
          },
          {
            kind: 'table',
            columns: ['Model', 'Published', 'NEURAX', 'Error'],
            rows: [
              ['LLaMA-2 7B', '6.74 B', '6.63 B', '−1.7 %'],
              ['Mistral 7B', '7.24 B', '7.13 B', '−1.5 %'],
              ['Qwen2 7B', '7.62 B', '7.62 B', '−0.04 %'],
              ['GPT-2', '124 M', '124 M', '< 1 %'],
            ],
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'Routed models are currently under-counted',
            text:
              'A mixture-of-experts model imported from its config analyses to {-roughly 22 % below its published size-} — Mixtral 8x7B comes out near 36 B against a published 46.7 B. This is not an import problem: the built-in Mixtral template produces the same figure. It is how the mixture-of-experts blocks are accounted for once they reach the compiler. {-Treat MoE parameter counts, and the memory and cost figures derived from them, as a lower bound-} until this is fixed.',
          },
        ],
      },
      {
        id: 'import-neurax',
        title: 'Importing a NEURAX design',
        summary: 'Re-open an architecture exported as JSON.',
        blocks: [
          {
            kind: 'text',
            text:
              'The same Import dialog accepts a NEURAX design — the JSON produced by **Export**. It is recognised automatically; you do not choose a format.',
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Import or Open?',
            text:
              '**Open** is for a `.neurax` document you saved, and restores everything including hardware settings and the last analysis. **Import** is for bringing in a model from elsewhere. If you exported a design and want it back exactly as it was, save it as a `.neurax` file instead and use Open.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'documents',
    title: 'Saving your work',
    sections: [
      {
        id: 'neurax-files',
        title: 'The .neurax file',
        summary: 'Your design as a document you own, on disk.',
        blocks: [
          {
            kind: 'text',
            text:
              'A design saves to a `.neurax` file: a plain, readable JSON document holding the whole design — every block and connection, the groups, the architecture family, the full hardware and training configuration, and the last analysis for reference.',
          },
          {
            kind: 'keys',
            items: [
              { keys: 'Ctrl+S', action: 'Save to the file this design came from' },
              { keys: 'Ctrl+Shift+S', action: 'Save As — choose a new file' },
              { keys: 'Ctrl+O', action: 'Open a .neurax file' },
            ],
          },
          {
            kind: 'text',
            text:
              'A dot beside **Save** in the toolbar means there are changes not yet written to the file.',
          },
          { kind: 'heading', text: 'Why a file and not just a project' },
          {
            kind: 'text',
            text:
              'Because a file can be committed next to the training code it describes, attached to a review, diffed against last week, and handed to a colleague who then opens it. The format is built for that: it is pretty-printed with a stable key order, so {+changing one hyperparameter changes exactly one line in `git diff`+} rather than rewriting the whole file.',
          },
          {
            kind: 'code',
            caption: 'A .neurax file opens with what it is',
            text: `{
  "format": "neurax-design",
  "version": 1,
  "generator": "NEURAX Studio",
  "savedAt": "2026-08-14T10:00:00.000Z",
  "name": "My LLaMA",
  "architecture": "transformer",
  "design": { "nodes": [...], "connections": [...], "groups": [...] },
  "hardware": { ... },
  "analysis": { ... }
}`,
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'The analysis in the file is a record, not an input',
            text:
              'It is kept so you can see yesterday\'s numbers without recomputing. Opening a file never reads it back as truth — press Run Analysis to recompute.',
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'In a browser, Save always asks',
            text:
              '{-A web page cannot write to a path-}, so every save in the browser build downloads a fresh copy. {+Only the desktop application can re-save a file in place+}, and only for files you opened or saved during that session.',
          },
        ],
      },
      {
        id: 'projects',
        title: 'Projects',
        summary: 'Designs the application keeps for you.',
        blocks: [
          {
            kind: 'text',
            text:
              'Alongside files, NEURAX keeps a list of saved projects, reachable from the toolbar. A project is stored by the application rather than by you, and survives restarts — on the desktop it is written to your profile directory and saved again when you quit.',
          },
          {
            kind: 'table',
            columns: ['', 'Project', '.neurax file'],
            rows: [
              ['Where it lives', 'Application profile', 'Wherever you put it'],
              ['Survives a restart', 'Yes', 'Yes'],
              ['Can go in git', 'No', 'Yes'],
              ['Can be sent to someone', 'No', 'Yes'],
              ['Best for', 'Work in progress', 'Anything you want to keep or share'],
            ],
          },
        ],
      },
      {
        id: 'exporting',
        title: 'Exporting',
        summary: 'Getting the architecture out in other forms.',
        blocks: [
          {
            kind: 'table',
            columns: ['Format', 'What it is'],
            rows: [
              ['**JSON**', 'The architecture and its analysis, re-importable into NEURAX.'],
              ['**NEURAX IR**', 'The exact topology the compiler analysed — the input to the analysis, useful for scripting or for filing a bug.'],
              ['**GitHub**', 'Push the architecture straight to a repository.'],
            ],
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'There is no PyTorch export',
            text:
              '{-There is no PyTorch or JAX export.-} There used to be framework emitters, and they were removed rather than repaired: they produced a class whose constructor was empty and whose forward pass was a chain of `x2 = x1` — an identity function under the model\'s name. Emitting nothing is better than emitting that. What leaves NEURAX today is {+the architecture itself, which it can describe truthfully+}.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'iterating',
    title: 'Working day to day',
    sections: [
      {
        id: 'undo',
        title: 'Undo and redo',
        summary: 'Nothing you do on the canvas is permanent.',
        blocks: [
          {
            kind: 'keys',
            items: [
              { keys: 'Ctrl+Z', action: 'Undo' },
              { keys: 'Ctrl+Shift+Z', action: 'Redo' },
              { keys: 'Ctrl+Y', action: 'Redo (Windows convention)' },
            ],
          },
          {
            kind: 'text',
            text:
              'History covers {+every change to the design+}: blocks added, moved, deleted or duplicated, connections made and broken, parameters edited, groups formed. A drag counts as one step rather than one per frame, so a single Ctrl+Z returns the block to where it started. {+Two hundred steps are kept.+}',
          },
          {
            kind: 'text',
            text:
              'Opening a file, loading a template or starting a blank page begins a new history — undo will not reach back into the document you left.',
          },
        ],
      },
      {
        id: 'compare',
        title: 'Comparing two designs',
        summary: 'Hold a baseline and see exactly what your change moved.',
        blocks: [
          {
            kind: 'text',
            text:
              'Design work is comparative. "Is 47 GB of VRAM good" has no answer; "is 32 layers at width 4096 better than 48 at 3072, for what it costs" does.',
          },
          {
            kind: 'steps',
            items: [
              'Analyse a design you want to measure against.',
              'Press **Compare**, then **Capture current design**.',
              'Close the panel, change the architecture, and run the analysis again.',
              'Reopen **Compare**. Both are shown side by side with the change in each metric.',
            ],
          },
          {
            kind: 'text',
            text:
              'Changes are coloured by whether they are an improvement, not by their sign — less VRAM is green, less throughput is red. Parameter count is left uncoloured, because more parameters is neither good nor bad. A change under half a percent shows as `=` rather than being dressed up as a result.',
          },
          {
            kind: 'note',
            tone: 'warning',
            title: 'It tells you when a comparison is not fair',
            text:
              'If the two were analysed on different GPUs, at a different precision, batch size, sequence length or architecture family, an amber box lists exactly which terms differ. The comparison still works — but {-part of the difference is the settings, not the architecture-}, and reading it otherwise is a mistake worth interrupting.',
          },
        ],
      },
      {
        id: 'shortcuts',
        title: 'Keyboard shortcuts',
        summary: 'Everything bound to a key.',
        blocks: [
          { kind: 'heading', text: 'Document' },
          {
            kind: 'keys',
            items: [
              { keys: 'Ctrl+S', action: 'Save' },
              { keys: 'Ctrl+Shift+S', action: 'Save As' },
              { keys: 'Ctrl+O', action: 'Open a .neurax file' },
            ],
          },
          { kind: 'heading', text: 'Help' },
          {
            kind: 'keys',
            items: [
              { keys: 'F1', action: 'Open this guide — works even while typing' },
            ],
          },
          { kind: 'heading', text: 'Editing' },
          {
            kind: 'keys',
            items: [
              { keys: 'Ctrl+Z', action: 'Undo' },
              { keys: 'Ctrl+Shift+Z  /  Ctrl+Y', action: 'Redo' },
              { keys: 'Ctrl+A', action: 'Select every block' },
              { keys: 'Ctrl+D', action: 'Duplicate the selection' },
              { keys: 'Ctrl+G', action: 'Group the selected blocks' },
              { keys: 'Delete  /  Backspace', action: 'Delete the selection' },
              { keys: 'Escape', action: 'Clear the selection, cancel a connection' },
            ],
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'Typing always wins',
            text:
              'While the cursor is in a text field, these shortcuts stand down — Ctrl+Z there undoes your typing, as it should.',
          },
          {
            kind: 'text',
            text: 'On macOS, use **Cmd** wherever **Ctrl** is written.',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'results',
    title: 'Reading the results',
    sections: [
      {
        id: 'metrics',
        title: 'What the numbers mean',
        summary: 'The metrics that decide whether a design is viable.',
        blocks: [
          { kind: 'heading', text: 'Model' },
          {
            kind: 'table',
            columns: ['Metric', 'Meaning'],
            rows: [
              ['Parameters', 'Total learnable weights. The headline size of the model.'],
              ['Layers', 'Depth, taken from the layer stack — not the number of blocks drawn.'],
            ],
          },
          { kind: 'heading', text: 'Memory' },
          {
            kind: 'table',
            columns: ['Metric', 'Meaning'],
            rows: [
              ['Peak VRAM', 'The high-water mark during a training step. This is the figure that decides whether the model fits.'],
              ['Weights', 'Parameters at the chosen precision.'],
              ['Activations', 'Intermediate tensors held for the backward pass. Gradient checkpointing trades these for compute.'],
              ['Gradients / Optimizer state', 'One gradient per parameter; AdamW adds two more values per parameter on top.'],
              ['Max batch that fits', 'The largest batch size that stays inside the target GPU memory.'],
            ],
          },
          { kind: 'heading', text: 'Compute' },
          {
            kind: 'table',
            columns: ['Metric', 'Meaning'],
            rows: [
              ['Total FLOPs', 'Floating-point operations for a full step, forward and backward.'],
              ['FLOPs per token', 'The usual scale for comparing language models.'],
              ['Arithmetic intensity', 'FLOPs per byte moved. Low means memory-bound, high means compute-bound.'],
              ['Roofline position', 'Where the model sits against the target chip\'s ridge point.'],
            ],
          },
          { kind: 'heading', text: 'Cost' },
          {
            kind: 'table',
            columns: ['Metric', 'Meaning'],
            rows: [
              ['Training cost', 'Dollars for a full run at the configured epochs and dataset size.'],
              ['Training time', 'Wall-clock hours on the configured GPU count.'],
              ['Energy / CO₂', 'Kilowatt-hours and kilograms of CO₂ for the run.'],
            ],
          },
          {
            kind: 'text',
            text:
              'The compiler produces **66 metrics** in total; the panels show the ones that matter for the view you are in. The full set is in the exported report.',
          },
        ],
      },
      {
        id: 'warnings',
        title: 'Warnings and diagnostics',
        summary: 'What NEURAX tells you without being asked.',
        blocks: [
          {
            kind: 'text',
            text:
              'The analysis reports problems it finds in the design as it goes: shapes that cannot be reconciled, a model that cannot fit the target, layers at risk of numerical instability at the chosen precision, a routing configuration that will load experts unevenly.',
          },
          {
            kind: 'text',
            text:
              'Warnings appear in the Architecture tab of the analysis panel and are counted in the inspector. Click through from the inspector to jump to them.',
          },
        ],
      },
      {
        id: 'accuracy',
        title: 'Accuracy and limits',
        summary: 'What is measured, what is estimated, and what is not verified.',
        blocks: [
          {
            kind: 'text',
            text:
              'NEURAX predicts. Some of what it predicts is checked against reality on every build, and some is not. Knowing which is which is part of using it well.',
          },
          { kind: 'heading', text: 'Verified against published figures' },
          {
            kind: 'text',
            text:
              '{+Parameter counts for reference models are asserted against their published values+} by the test suite on every build. Seven models are checked in the Rust core, and four imported configs are checked {+end to end through the real compiler+}.',
          },
          {
            kind: 'table',
            columns: ['Model', 'Published', 'NEURAX', 'Error'],
            rows: [
              ['VGG-16', '138.0 M', '138.4 M', '+0.3 %'],
              ['Mixtral 8x7B (core)', '46.7 B', '47.4 B', '+1.5 %'],
              ['LLaMA-2 70B', '70.0 B', '68.7 B', '−1.8 %'],
              ['ResNet-50', '25.6 M', '26.5 M', '+3.5 %'],
              ['RWKV 7B', '7.5 B', '7.2 B', '−4.2 %'],
              ['DeepSeek-V3', '671 B', '701 B', '+4.5 %'],
              ['Mamba 2.8B', '2.80 B', '2.66 B', '−4.9 %'],
            ],
          },
          {
            kind: 'text',
            text:
              'Four of those were badly wrong before that test existed — Mixtral by +122 %, DeepSeek by +108 %, RWKV by −96.7 %, LLaMA-2 by −28.7 % — because nothing compared a computed figure to a known one. Seven models is what is measured. It is not a claim about every architecture that exists.',
          },
          { kind: 'heading', text: 'Not verified against measurement' },
          {
            kind: 'list',
            items: [
              '**Latency and throughput** are roofline estimates from the target chip\'s specifications. {-No measured run has been compared against them.-}',
              '**Training cost and time** follow from those estimates and inherit their uncertainty, plus whatever your cloud actually charges.',
              '**Energy and CO₂** rest on published chip power figures and an assumed grid intensity.',
              '**Inference Intelligence** widgets — stability, entropy, hallucination risk — are analytical indicators, not measurements of a served model.',
            ],
          },
          { kind: 'heading', text: 'Known gaps' },
          {
            kind: 'list',
            items: [
              '**Mixture-of-experts models imported or drawn in the studio {-under-count by roughly 22 %-}** against their published size. The compiler\'s own MoE path (the figure in the table above) is accurate; the studio path is not. {-Treat MoE numbers from the canvas as a lower bound.-}',
              '**Multimodal models** import their language tower only. {-Vision and audio towers are ignored-}, so totals for those models are the text side alone.',
              '**No calibration loop.** {-NEURAX does not learn from your measured runs.-} Every prediction is the formula\'s, not your cluster\'s.',
            ],
          },
          {
            kind: 'note',
            tone: 'info',
            title: 'The honest summary',
            text:
              '{+Trust the structural figures — parameters, memory, FLOPs — for dense models, within a few percent.+} {-Treat latency, cost and carbon as estimates for comparing designs against each other, not as forecasts of your invoice.-}',
          },
        ],
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════
  {
    id: 'troubleshooting',
    title: 'Troubleshooting',
    sections: [
      {
        id: 'common-problems',
        title: 'Common problems',
        summary: 'What to check when something looks wrong.',
        blocks: [
          { kind: 'heading', text: 'The parameter count is far too low' },
          {
            kind: 'text',
            text:
              'Almost always the layer stack. Check that the blocks forming one layer body are connected back to the `layer_stack` block, and that its `num_layers` is right. A body that is not connected back is counted once instead of once per layer.',
          },
          { kind: 'heading', text: 'A block seems to be ignored' },
          {
            kind: 'text',
            text:
              'Check it is connected at both ends. A block with no path from the input is not part of the model, and the analysis will say so in its warnings.',
          },
          { kind: 'heading', text: 'The analysis will not run' },
          {
            kind: 'text',
            text:
              'The design needs at least one input and one output, and a path between them. If the compiler cannot resolve a tensor shape it reports which block and why, in the Architecture tab.',
          },
          { kind: 'heading', text: 'Two analyses of the same design differ' },
          {
            kind: 'text',
            text:
              'They should not — the compiler is deterministic. If they do, something changed in the hyperparameters or the target between the two runs. The Compare panel will show you exactly which setting moved.',
          },
          { kind: 'heading', text: 'A file will not open' },
          {
            kind: 'text',
            text:
              '**Open** takes `.neurax` documents only. An exported NEURAX IR or a HuggingFace config goes through **Import** instead. If the file was written by a newer version of NEURAX, it will say so rather than opening it and dropping what it cannot read.',
          },
          { kind: 'heading', text: 'Save says it cannot write in place' },
          {
            kind: 'text',
            text:
              'The desktop application only overwrites files you opened or saved during the current session, and the file may have been moved or made read-only since. It falls back to Save As so you can choose a new location.',
          },
        ],
      },
      {
        id: 'where-things-live',
        title: 'Where NEURAX keeps things',
        summary: 'Files on disk, on each platform.',
        blocks: [
          {
            kind: 'text',
            text: 'Saved projects live in your profile directory, under `dev.neurax.desktop`:',
          },
          {
            kind: 'table',
            columns: ['Platform', 'Location'],
            rows: [
              ['Linux', '`~/.local/share/dev.neurax.desktop/projects.json`'],
              ['macOS', '`~/Library/Application Support/dev.neurax.desktop/projects.json`'],
              ['Windows', '`%APPDATA%\\dev.neurax.desktop\\projects.json`'],
            ],
          },
          {
            kind: 'text',
            text:
              'Your `.neurax` files live wherever you saved them. API keys for the Copilot are kept in the application profile and never travel with a design.',
          },
        ],
      },
    ],
  },
];

/** Every section, flattened, for search and for direct linking. */
export function allSections(): Array<DocSection & { chapter: string; chapterId: string }> {
  return DOCUMENTATION.flatMap((chapter) =>
    chapter.sections.map((section) => ({
      ...section,
      chapter: chapter.title,
      chapterId: chapter.id,
    })),
  );
}

/** The searchable text of a section, title and body together. */
function searchableText(section: DocSection): string {
  const parts: string[] = [section.title, section.summary];

  for (const block of section.blocks) {
    switch (block.kind) {
      case 'text':
      case 'heading':
        parts.push(block.text);
        break;
      case 'code':
        parts.push(block.text, block.caption ?? '');
        break;
      case 'list':
      case 'steps':
        parts.push(...block.items);
        break;
      case 'keys':
        parts.push(...block.items.map((k) => `${k.keys} ${k.action}`));
        break;
      case 'table':
        parts.push(...block.columns, ...block.rows.flat());
        break;
      case 'note':
        parts.push(block.title, block.text);
        break;
    }
  }

  return parts.join(' ').toLowerCase();
}

/**
 * Sections matching a query, best first.
 *
 * Deliberately simple: every term must appear somewhere in the section, and
 * sections whose title matches sort first. A guide this size does not need an
 * index, and a fuzzy matcher would return the wrong section confidently rather
 * than nothing honestly.
 */
export function searchDocs(query: string): Array<DocSection & { chapter: string; chapterId: string }> {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [];

  return allSections()
    .map((section) => {
      const haystack = searchableText(section);
      if (!terms.every((term) => haystack.includes(term))) return null;

      const title = section.title.toLowerCase();
      const score = terms.filter((term) => title.includes(term)).length;
      return { section, score };
    })
    .filter((hit): hit is { section: ReturnType<typeof allSections>[number]; score: number } => hit !== null)
    .sort((a, b) => b.score - a.score)
    .map((hit) => hit.section);
}

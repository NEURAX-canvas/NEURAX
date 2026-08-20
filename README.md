# NEURAX

**Know what a model costs — before you spend a single GPU-hour finding out.**

The usual way to find out an architecture doesn't fit is to launch the training run and watch it OOM six hours in  or finish, and cost three times what you budgeted, because nothing checked before you committed the GPU-hours. NEURAX checks first. Point it at an architecture  a full LLM or a small model built for your own dataset  and it hands back memory, speed, and training cost in under 50 milliseconds. No GPU, no training run, no waiting to find out.

[![CI](https://github.com/rustnew/NEURAX/actions/workflows/ci.yml/badge.svg)](https://github.com/rustnew/NEURAX/actions)
[![Release](https://img.shields.io/github/v/release/rustnew/NEURAX?style=flat-square&color=blue)](https://github.com/rustnew/NEURAX/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<p align="center">
  <img width="820" alt="A 28B-parameter transformer (GQA, SwiGLU) loaded on the canvas, with real FLOPs, VRAM, and compute-efficiency numbers computed live" src="https://github.com/user-attachments/assets/d4324dc6-92a1-4742-a849-aa21d1b149ca" />
  <br><sub>A 28B GQA/SwiGLU transformer, loaded and analyzed — parameters, FLOPs, VRAM, and roofline position, all computed the moment it was built.</sub>
</p>

## Install the desktop app

```bash
curl -fsSL https://raw.githubusercontent.com/rustnew/NEURAX/main/install.sh | sh
```

Then type `neurax`, or open **NEURAX** from your applications menu. That's the entire install — no `sudo`, nothing written outside your home directory, nothing to sign up for.

<p align="center">
  <img width="600" alt="A real terminal running the install command above, on a real machine" src="https://github.com/user-attachments/assets/e30d2ff1-2fe9-4573-9f84-ac17c858b468" />
  <br><sub>The command above, actually run — download, install, and NEURAX ready to open, in one pass.</sub>
</p>

Runs on **Debian, Ubuntu, Kali, Arch, Fedora, and effectively any Linux**, plus **macOS** (Intel and Apple silicon).

Not sure yet? Read the install script before running it — one file, plain shell: [`install.sh`](install.sh).

## What it gives you

Drag blocks onto a canvas, or load one of 88 real reference architectures — LLM, diffusion, MoE, and more — and get:

- **Will it fit?** Peak VRAM, activations, optimizer state — to the byte.
- **What will it cost?** Training time, dollars, energy, CO₂ — on the hardware you actually have.
- **Where's the bottleneck?** FLOPs by operation, per layer, memory- or compute-bound.
- **Will it behave?** Inference stability and hallucination risk, before you serve a token.

Every number is computed from the architecture you built, live — not a lookup table.

<p align="center">
  <img width="49%" alt="Importing a model directly from a HuggingFace config.json, by URL or by pasting the file" src="https://github.com/user-attachments/assets/ba4445ef-d13f-4a7f-874c-a691b3cd891c" />
  <img width="49%" alt="A Mixture-of-Experts diffusion model (MMDiT blocks, VAE, T5-XXL text encoder) analyzed on the canvas" src="https://github.com/user-attachments/assets/69771bb1-9954-4dfd-a7a2-ed18f8c39e69" />
  <br><sub>Left: import any public model straight from HuggingFace — paste an ID or a <code>config.json</code>, nothing uploaded. Right: the same compiler, on a diffusion architecture — not just LLMs.</sub>
</p>

**This is what the opening line means, in practice** — a 512-expert, 92-layer design pushed deliberately past a single GPU, caught before any training run:

<p align="center">
  <img width="820" alt="A design NEURAX flags as OOM Risk: CRITICAL — 21.5 TB of peak VRAM against an 80 GB GPU, found in milliseconds, not six hours into training" src="https://github.com/user-attachments/assets/a820b4d7-e317-4fc4-b6df-88f0f6b76704" />
  <br><sub>21.5 TB of peak VRAM against an 80 GB card — NEURAX says so instantly, not partway through a training run that was never going to finish.</sub>
</p>

<p align="center">
  <img width="820" alt="The Time Machine's regulatory compliance timeline — EU AI Act, CSRD, and other real, dated obligations checked automatically" src="https://github.com/user-attachments/assets/f91bb600-8e5d-4cea-a00a-73ae88926eb3" />
  <br><sub>Time Machine also tracks what training that model would actually be obligated to — EU AI Act thresholds, carbon reporting, verified against real, dated regulation text, not a generic checklist.</sub>
</p>

<p align="center">
  <img width="820" alt="Comparing two designs side by side — parameters, VRAM, FLOPs, and latency, with the percentage change for each" src="https://github.com/user-attachments/assets/91e7403a-a634-4e13-8eb2-83c6eea87a52" />
  <br><sub>Change an architecture decision and see exactly what moved — every metric, as a percentage, not just a new number to compare by eye.</sub>
</p>

**This isn't only for people who already have a research cluster.** A small, specialized model tuned to your own dataset gets the same precise answer as a 70B one. That's the common case NEURAX is built for.

## It doesn't ask for anything

- No account to create — a local profile is generated automatically, kept on your machine.
- No API key leaves your browser — the AI copilot (bring your own key: OpenAI, Anthropic, Gemini, Mistral, Fireworks, DeepSeek, GLM, or any custom endpoint) talks straight to your chosen provider.
- No project ever uploaded anywhere. The compiler runs on your machine; that's the whole design.

## Accuracy, measured

Every reference model is checked against its real published size by an automated test — not eyeballed once:

| Model | Published | NEURAX | Error |
|---|---|---|---|
| Mixtral 8x7B | 46.7 B | 47.4 B | +1.5 % |
| LLaMA-2 70B | 70.0 B | 68.7 B | −1.8 % |
| DeepSeek-V3 | 671 B | 701 B | +4.5 % |
| Mamba 2.8B | 2.80 B | 2.66 B | −4.9 % |

Four of these were wrong before that test existed — one by +122%. That's why the test exists, and why it still runs on every change.

## Learn more

The one-liner above is the reference way to run NEURAX — the desktop app, with the compiler embedded. For everything else — Windows and web builds, Docker, running it as a Rust library, the full architecture, the API reference — see the **[documentation](https://rustnew.github.io/NEURAX/)**.

[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Security](SECURITY.md) · [License: MIT](LICENSE)

---

Built by Fossouo.

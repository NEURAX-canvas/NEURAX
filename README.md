<img width="1919" height="918" alt="Capture d’écran du 2026-08-16 12-37-49" src="https://github.com/user-attachments/assets/e30d2ff1-2fe9-4573-9f84-ac17c858b468" />

# NEURAX

**Know what a model costs — before you spend a single GPU-hour finding out.**

The usual way to find out an architecture doesn't fit is to launch the training run and watch it OOM six hours in — or finish, and cost three times what you budgeted, because nothing checked before you committed the GPU-hours. NEURAX checks first. Point it at an architecture — a full LLM or a small model built for your own dataset — and it hands back memory, speed, and training cost in under 50 milliseconds. No GPU, no training run, no waiting to find out.

[![CI](https://github.com/rustnew/NEURAX/actions/workflows/ci.yml/badge.svg)](https://github.com/rustnew/NEURAX/actions)
[![Release](https://img.shields.io/github/v/release/rustnew/NEURAX?style=flat-square&color=blue)](https://github.com/rustnew/NEURAX/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<p align="center">
  <img width="1916" height="1075" alt="Capture d’écran du 2026-08-16 08-59-57" src="https://github.com/user-attachments/assets/69771bb1-9954-4dfd-a7a2-ed18f8c39e69" />
  <img width="1906" height="1076" alt="Capture d’écran du 2026-08-14 13-01-49" src="https://github.com/user-attachments/assets/cc7d4f27-4276-49cd-bd79-e3e695e9b791" />
  <img width="1920" height="1077" alt="Capture d’écran du 2026-08-18 08-09-52" src="https://github.com/user-attachments/assets/96a4fb4e-7a30-474a-9816-e1e783510ae1" />
  <img width="1920" height="1077" alt="Capture d’écran du 2026-08-18 08-04-14" src="https://github.com/user-attachments/assets/d4324dc6-92a1-4742-a849-aa21d1b149ca" />
  <img width="1920" height="1077" alt="Capture d’écran du 2026-08-18 08-00-28" src="https://github.com/user-attachments/assets/91e7403a-a634-4e13-8eb2-83c6eea87a52" />
<img width="1920" height="1077" alt="Capture d’écran du 2026-08-18 07-50-10" src="https://github.com/user-attachments/assets/a820b4d7-e317-4fc4-b6df-88f0f6b76704" />
<img width="1920" height="1077" alt="Capture d’écran du 2026-08-18 07-49-12" src="https://github.com/user-attachments/assets/ba4445ef-d13f-4a7f-874c-a691b3cd891c" />
<img width="1916" height="1075" alt="Capture d’écran du 2026-08-16 09-07-32" src="https://github.com/user-attachments/assets/d7c8b293-a2b8-4a38-af02-d3490a40bb1f" />
<img width="1916" height="1075" alt="Capture d’écran du 2026-08-16 09-00-55" src="https://github.com/user-attachments/assets/567474d4-051a-4b37-97dd-5a7de3e963b3" />
</p>

## Install it now

```bash
curl -fsSL https://raw.githubusercontent.com/rustnew/NEURAX/main/install.sh | sh
```

Then type `neurax`, or open **NEURAX** from your applications menu. That's the entire install — no `sudo`, nothing written outside your home directory, nothing to sign up for.

Runs on **Debian, Ubuntu, Kali, Arch, Fedora, and effectively any Linux**, plus **macOS** (Intel and Apple silicon). Verified live, not assumed: while writing this, a real run of this exact command downloaded the app, installed it, and launched it for real.

Not sure yet? Read the install script before running it — one file, plain shell: [`install.sh`](install.sh).

## What it gives you

<p align="center">
 <img width="1920" height="1077" alt="Capture d’écran du 2026-08-18 07-46-47" src="https://github.com/user-attachments/assets/f91bb600-8e5d-4cea-a00a-73ae88926eb3" />
 <img width="1920" height="1077" alt="Capture d’écran du 2026-08-18 07-47-31" src="https://github.com/user-attachments/assets/a0c6631e-ee29-4edd-a3a6-01dcf13648ee" />


</p>

Drag blocks onto a canvas, or load one of 88 real reference architectures, and get:

- **Will it fit?** Peak VRAM, activations, optimizer state — to the byte.
- **What will it cost?** Training time, dollars, energy, CO₂ — on the hardware you actually have.
- **Where's the bottleneck?** FLOPs by operation, per layer, memory- or compute-bound.
- **Will it behave?** Inference stability and hallucination risk, before you serve a token.

Every number is computed from the architecture you built, live — not a lookup table.

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

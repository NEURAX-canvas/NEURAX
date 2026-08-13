# NEURAX Desktop

The NEURAX studio as an application you install, with the compiler running
inside it.

Everything the web version does, it does — same interface, same analyses, same
numbers — with one difference that matters: there is no server. The Rust
compiler that produces the analysis is linked into the application and answers
on a loopback socket. Nothing you design is uploaded, no account is required,
and it works with the network unplugged.

---

## Installing

Download the installer for your platform from
[Releases](https://github.com/rustnew/NEURAX/releases):

| Platform | File |
|---|---|
| Linux (Debian/Ubuntu) | `neurax_<version>_amd64.deb` |
| Linux (Fedora/RHEL) | `neurax-<version>.x86_64.rpm` |
| Linux (any) | `neurax_<version>_amd64.AppImage` |
| macOS | `NEURAX_<version>_universal.dmg` |
| Windows | `NEURAX_<version>_x64-setup.exe` |

Then launch it from your applications menu, or type `neurax` in a terminal.

### `neurax` on the command line

The `neurax` command is the CLI compiler, and it keeps every subcommand it
had. Run it with no arguments and it opens the desktop window instead of
printing usage:

```
neurax              # opens the application
neurax gui          # the same thing, explicitly
neurax analyze model.json    # unchanged
neurax compile model.json    # unchanged
```

---

## Building from source

You need the Rust toolchain, Node 20+, and the Tauri CLI:

```bash
cargo install tauri-cli --version '^2'
```

**Linux also needs the webview development packages.** Tauri renders through
the operating system's webview rather than shipping its own browser, which is
why the binary is ~15 MB instead of ~150 MB, and why these are required:

```bash
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev \
                 libayatana-appindicator3-dev librsvg2-dev \
                 build-essential curl wget file libssl-dev
```

On Fedora: `webkit2gtk4.1-devel gtk3-devel libappindicator-gtk3-devel
librsvg2-devel`. macOS needs Xcode command line tools; Windows needs the
WebView2 runtime, which Windows 11 already has and the installer adds
otherwise.

Then:

```bash
cd neurax-desktop
cargo tauri dev      # run it, with the UI hot-reloading
cargo tauri build    # produce installers in target/release/bundle/
```

`cargo tauri build` runs `npm run build:desktop` in `neurax-ui` first, so the
frontend does not need building separately.

---

## How it is put together

```
   ┌─ neurax-desktop ─────────────────────────────────────┐
   │                                                       │
   │   Tauri window                                        │
   │     └── neurax-ui, built to static files and          │
   │         loaded from the bundle (tauri://localhost)    │
   │                        │                              │
   │                        │  HTTP, on 127.0.0.1:<port>   │
   │                        ▼                              │
   │   neurax-service, as a library, in this process       │
   │     └── neurax-core → the 10-pass IR pipeline         │
   │                                                       │
   └───────────────────────────────────────────────────────┘
```

Three decisions are worth stating, because each was a choice between options
that both work.

**The frontend is shared, not copied.** `neurax-ui` is one codebase serving
both the website and this application. The only thing it does differently here
is read its API address from a value the host injects at launch instead of one
baked in at build time — see `src/services/desktopRuntime.ts`. A copy would
have been quicker and would have started drifting within weeks.

**The API is the real API, mounted in-process.** `neurax-service` was split
into a library and a thin binary; both mount the same routing table from
`configure_routes`. The desktop app is not a reimplementation of the service,
so an endpoint written for the web app exists here the moment it is written.

**The port is bound before the window opens.** The socket is bound on
`127.0.0.1:0`, the OS assigns a port, and that port goes into the page's
bootstrap script. There is no fixed port to collide with something else you are
running, no retry loop, and no "connecting to backend" state — by the time the
UI exists, its API is already listening.

The API is bound to loopback, never `0.0.0.0`. Nothing NEURAX analyses is
reachable from the network.

---

## Regenerating the icons

The icons are rendered from `neurax-ui/public/neurax-favicon.svg`, the same
mark the website serves:

```bash
cd icons
npm install sharp
node generate.mjs     # PNGs, and a 1024px master
python3 icon.py       # icon.icns and icon.ico from that master
```

The results are committed, so this only needs running when the brand mark
changes.

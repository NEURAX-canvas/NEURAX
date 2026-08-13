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

On Linux and macOS, one command:

```bash
curl -fsSL https://raw.githubusercontent.com/rustnew/NEURAX/main/install.sh | sh
```

It downloads the bundle for your platform, installs it under `~/.local`, adds
it to your applications menu, and makes `neurax` available in the shell.
Nothing is written outside your home directory and no step runs under sudo.

On Linux it takes the AppImage when one was published, and otherwise unpacks
the `.deb` or the `.rpm` into your home directory rather than handing it to a
package manager — same files, no root. That fallback is not theoretical: the
AppImage bundler downloads tooling at build time, and a 503 from that download
is enough to produce a release with a `.deb` and an `.rpm` and no AppImage.

    --version <tag>   install a specific release
    --prefix <dir>    install somewhere other than ~/.local
    --uninstall       remove it again

On Windows, or to install by hand, take a file from
[Releases](https://github.com/rustnew/NEURAX/releases):

| Platform | File |
|---|---|
| Linux, any distribution | `NEURAX_<version>_amd64.AppImage` |
| Debian, Ubuntu | `NEURAX_<version>_amd64.deb` |
| Fedora, RHEL | `NEURAX-<version>.x86_64.rpm` |
| macOS, universal | `NEURAX_<version>_universal.dmg` |
| Windows | `NEURAX_<version>_x64-setup.exe` |

Then launch it from your applications menu, or type `neurax` in a terminal.

### Publishing a release

`Build desktop installers` runs on a `v*` tag and creates a **draft** release.
The public releases API does not show drafts to anonymous callers, so
`install.sh` will not find it until the draft is published — which means a
build nobody checked is never installable. Run the workflow by hand
(`workflow_dispatch`) to produce the same bundles as downloadable artifacts
without publishing anything.

### `neurax` on the command line

The installer puts the application on your PATH as `neurax`, so typing it opens
the window:

```
neurax              # opens NEURAX
neurax-desktop      # the same binary, under its own name
```

There is no CLI compiler any more. The crate that provided `neurax analyze`
and `neurax compile` has been removed; for analysis without a window, run
`neurax-service` and call `/analyze` over HTTP.

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

## Identical to the web application — and checked

Someone should be able to move between the website and this application
without noticing. That is not a promise the code makes on its own, so it is
tested from both ends.

**Same interface, structurally.** There is one frontend, not two. Host
detection is allowed to decide *how* something happens — which file dialog
opens, which route `/` lands on — never *what exists*.
`src/services/hostParity.test.ts` fails if any file outside a short, reasoned
list branches on the host, if a desktop-only component appears, or if the
Tauri API is imported statically into the web bundle.

**Same capabilities, provably.** `neurax-service/tests/desktop_parity.rs`
starts the API the way this application starts it — loopback listener,
`tauri://` origins, no Supabase — and then calls every endpoint the frontend
actually uses: analysis, the hardware database, presets, inference simulation,
the time machine, hardware comparison, compliance, credits, and the full
project lifecycle. It also checks that CORS admits `tauri://localhost` and
refuses an arbitrary web origin, because a preflight rejection would break
every request without any handler ever being reached.

### Where the desktop is better

| | Web | Desktop |
|---|---|---|
| Analysis | Round trip to a server | In-process, no network |
| Works offline | No | Entirely |
| Account | Required | None |
| Your designs | Sent to a server | Never leave the machine |
| Projects after a restart | Lost | Kept, in your data directory |
| Exporting | Browser download, location chosen for you | System save dialog, real path reported |

Project persistence lives in `neurax_service::persistence`, not in this crate,
so a self-hosted deployment of the web service gets it too by setting
`NEURAX_PROJECTS_FILE`. The behaviour is shared rather than duplicated,
which is the only way the two can be relied on to stay the same.

### Window controls

The title bar with its minimise, maximise and close buttons is the platform's
own on macOS and Windows, where replacing it would make NEURAX the one window
on the machine that behaves differently.

On Linux it is drawn by the application. That is not a preference: on the
desktop this was tested against, the window came back with
`_NET_FRAME_EXTENTS = 0, 0, 0, 0` — no frame at all — while
`_NET_WM_ALLOWED_ACTIONS` still listed minimise, maximise and close. The window
was fully controllable and there was simply nothing to click. The decision
lives in one constant, `OWN_TITLE_BAR` in `src/main.rs`, and travels to the
frontend in the same bootstrap script that carries the API address.

### The one deliberate difference

On the web, `/` is the landing page. In the desktop application, once you have
a profile, `/` opens the studio directly — someone who has already installed
NEURAX does not need to be sold it. Before that it still shows the landing
page, because that is where the profile is created.

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

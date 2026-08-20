# Desktop App

The NEURAX studio as an application you install, with the compiler running
inside it — same interface, same analyses, same numbers as the web version,
with one difference that matters: there is no server. Nothing you design is
uploaded, no account is required, and it works with the network unplugged.

Runs on Linux, macOS, and Windows.

## Linux and macOS

One command:

```bash
curl -fsSL https://raw.githubusercontent.com/rustnew/NEURAX/main/install.sh | sh
```

It downloads the bundle for your platform, installs it under `~/.local`, adds
it to your applications menu, and makes `neurax` available in the shell.
Nothing is written outside your home directory and no step runs under sudo.

```
--version <tag>   install a specific release instead of the newest
--prefix <dir>    install somewhere other than ~/.local
--uninstall       remove what the installer put there
```

Not sure yet? Read the script before running it — one file, plain POSIX
shell: [`install.sh`](https://github.com/rustnew/NEURAX/blob/main/install.sh).

## Windows

There is no one-line installer — `install.sh` is a POSIX shell script and
doesn't run under Windows' own shells. Take the installer from
[Releases](https://github.com/rustnew/NEURAX/releases) instead:

**`NEURAX_<version>_x64-setup.exe`**

Run it and launch NEURAX from the Start menu. Windows 11 already has the
WebView2 runtime NEURAX renders through; the installer adds it on Windows 10
if it's missing.

## Installing by hand, any platform

Every release publishes one asset per platform:

| Platform | File |
|---|---|
| Linux, any distribution | `NEURAX_<version>_amd64.AppImage` |
| Debian, Ubuntu | `NEURAX_<version>_amd64.deb` |
| Fedora, RHEL | `NEURAX-<version>.x86_64.rpm` |
| macOS, universal (Intel and Apple silicon) | `NEURAX_<version>_universal.dmg` |
| Windows | `NEURAX_<version>_x64-setup.exe` |

## Building from source, and how it's put together

Covered in the crate's own README, not duplicated here:
[`neurax-desktop/README.md`](https://github.com/rustnew/NEURAX/blob/main/neurax-desktop/README.md)
— toolchain requirements per platform, the Tauri build commands, how the
desktop app embeds the real `neurax-service` API in-process rather than
reimplementing it, and the test that checks the desktop and web builds stay
identical in what they can do.

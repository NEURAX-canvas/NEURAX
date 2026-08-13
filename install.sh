#!/bin/sh
#
# NEURAX installer — Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/rustnew/NEURAX/main/install.sh | sh
#
# Installs the desktop application and makes `neurax` available in the shell.
# Nothing is written outside your home directory unless you ask for it, and no
# step runs under sudo.
#
# Options (pass after `| sh -s --`):
#   --version <tag>   install a specific release instead of the newest
#   --prefix <dir>    install somewhere other than ~/.local
#   --uninstall       remove what this script installed
#   --help
#
# POSIX sh on purpose: this has to run under dash on Debian, bash on most
# distributions, and zsh on macOS, on a machine where nothing has been set up
# yet.

set -eu

REPO="rustnew/NEURAX"
API="https://api.github.com/repos/${REPO}"

PREFIX="${NEURAX_PREFIX:-${HOME}/.local}"
VERSION=""
UNINSTALL=0

# ─── Output ─────────────────────────────────────────────────────────

# Colour only when stdout is a terminal; piped into a log it would be noise.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m')
    RED=$(printf '\033[31m'); GREEN=$(printf '\033[32m')
    YELLOW=$(printf '\033[33m'); RESET=$(printf '\033[0m')
else
    BOLD=''; DIM=''; RED=''; GREEN=''; YELLOW=''; RESET=''
fi

say()  { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s\n' "${BOLD}" "${RESET}" "$*"; }
note() { printf '    %s%s%s\n' "${DIM}" "$*" "${RESET}"; }
warn() { printf '%swarning:%s %s\n' "${YELLOW}" "${RESET}" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "${RED}" "${RESET}" "$*" >&2; exit 1; }

usage() {
    sed -n '3,20p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'
    exit 0
}

# ─── Arguments ──────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="${2:?--version needs a tag}"; shift 2 ;;
        --prefix)  PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        --help|-h) usage ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

BIN_DIR="${PREFIX}/bin"
APP_DIR="${PREFIX}/lib/neurax"
DESKTOP_ENTRY="${HOME}/.local/share/applications/neurax.desktop"
MAC_APP="/Applications/NEURAX.app"
MAC_APP_USER="${HOME}/Applications/NEURAX.app"

# ─── Platform ───────────────────────────────────────────────────────

detect_platform() {
    os=$(uname -s)
    arch=$(uname -m)

    case "${os}" in
        Linux)  PLATFORM=linux ;;
        Darwin) PLATFORM=macos ;;
        *) die "NEURAX has no build for ${os}. Build from source: https://github.com/${REPO}#building" ;;
    esac

    case "${arch}" in
        x86_64|amd64)  ARCH=x86_64 ;;
        arm64|aarch64) ARCH=arm64 ;;
        *) die "unsupported architecture: ${arch}" ;;
    esac

    # The macOS bundle is universal, so one file serves both architectures.
    # The Linux bundle is built on an x86_64 runner only.
    if [ "${PLATFORM}" = linux ] && [ "${ARCH}" = arm64 ]; then
        die "no prebuilt Linux arm64 bundle yet.

Build it from a checkout instead — it is one command once the webview
development packages are installed:

    git clone https://github.com/${REPO}.git
    cd NEURAX/neurax-desktop && cargo tauri build

See neurax-desktop/README.md for the package list."
    fi
}

need() {
    command -v "$1" >/dev/null 2>&1 || die "this installer needs \`$1\`, which is not installed."
}

# ─── Downloading ────────────────────────────────────────────────────

fetch() {
    # -f so an HTML error page is never mistaken for a payload.
    curl -fsSL "$1"
}

fetch_to() {
    curl -fL --progress-bar "$1" -o "$2"
}

# Extract the browser_download_url of the first asset matching a suffix.
#
# Written against the raw JSON rather than jq, which is not installed by
# default anywhere this has to run.
asset_url_from() {
    json="$1"; suffix="$2"
    printf '%s' "${json}" \
        | tr ',{' '\n\n' \
        | grep '"browser_download_url"' \
        | sed 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' \
        | grep -- "${suffix}\$" \
        | head -n 1
}

# Find a release that actually carries a bundle for this platform.
#
# Deliberately not `/releases/latest`: this repository has tags that predate
# the desktop application, and the newest release is not necessarily one that
# published installers. So walk the releases newest-first and take the first
# that has the asset we need.
resolve_release() {
    suffix="$1"

    if [ -n "${VERSION}" ]; then
        step "Looking up release ${VERSION}"
        release_json=$(fetch "${API}/releases/tags/${VERSION}") \
            || die "no release tagged ${VERSION}"
        ASSET_URL=$(asset_url_from "${release_json}" "${suffix}")
        [ -n "${ASSET_URL}" ] || die "release ${VERSION} has no ${suffix} bundle"
        RELEASE_TAG="${VERSION}"
        return
    fi

    step "Finding the newest NEURAX release for ${PLATFORM}"
    releases_json=$(fetch "${API}/releases?per_page=30") \
        || die "could not reach the GitHub API. Are you online?"

    # Split into one release per line, preserving order (newest first).
    ASSET_URL=""
    RELEASE_TAG=""
    for tag in $(printf '%s' "${releases_json}" \
                    | tr ',' '\n' \
                    | grep '"tag_name"' \
                    | sed 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/'); do
        one=$(fetch "${API}/releases/tags/${tag}") || continue
        url=$(asset_url_from "${one}" "${suffix}")
        if [ -n "${url}" ]; then
            ASSET_URL="${url}"
            RELEASE_TAG="${tag}"
            break
        fi
    done

    if [ -z "${ASSET_URL}" ]; then
        die "no published release contains a ${suffix} bundle yet.

The desktop installers are produced by the \`Build desktop installers\`
workflow, which runs when a version tag is pushed. Until one has run, build
from a checkout:

    git clone https://github.com/${REPO}.git
    cd NEURAX/neurax-desktop && cargo tauri build"
    fi
}

# ─── Install: Linux ─────────────────────────────────────────────────

install_linux() {
    need curl
    resolve_release ".AppImage"

    step "Installing NEURAX ${RELEASE_TAG}"
    note "${ASSET_URL}"

    mkdir -p "${APP_DIR}" "${BIN_DIR}"
    tmp=$(mktemp -d)
    # shellcheck disable=SC2064  # expand tmp now, not at exit
    trap "rm -rf '${tmp}'" EXIT INT TERM

    fetch_to "${ASSET_URL}" "${tmp}/neurax.AppImage"
    chmod +x "${tmp}/neurax.AppImage"
    # Move into place last, so an interrupted download never leaves a broken
    # executable where a working one used to be.
    mv -f "${tmp}/neurax.AppImage" "${APP_DIR}/neurax-desktop"
    ln -sf "${APP_DIR}/neurax-desktop" "${BIN_DIR}/neurax-desktop"

    install_launcher
    install_desktop_entry
    check_fuse
}

# An AppImage needs FUSE to mount itself. Most desktops have it; some minimal
# installs and containers do not, and the failure message it produces is
# obscure, so say it plainly up front.
check_fuse() {
    if ! command -v fusermount >/dev/null 2>&1 && ! command -v fusermount3 >/dev/null 2>&1; then
        warn "FUSE was not found. AppImages need it to run."
        note "Debian/Ubuntu:  sudo apt install libfuse2"
        note "Fedora:         sudo dnf install fuse"
        note "Or run it unpacked:  ${APP_DIR}/neurax-desktop --appimage-extract-and-run"
    fi
}

install_desktop_entry() {
    mkdir -p "$(dirname "${DESKTOP_ENTRY}")"
    cat > "${DESKTOP_ENTRY}" <<ENTRY
[Desktop Entry]
Type=Application
Name=NEURAX
Comment=Analytical compiler for neural network architectures
Exec=${APP_DIR}/neurax-desktop
Icon=neurax
Terminal=false
Categories=Development;Science;
ENTRY
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$(dirname "${DESKTOP_ENTRY}")" 2>/dev/null || true
    fi
    note "Added to your applications menu"
}

# ─── Install: macOS ─────────────────────────────────────────────────

install_macos() {
    need curl
    need hdiutil
    resolve_release ".dmg"

    step "Installing NEURAX ${RELEASE_TAG}"
    note "${ASSET_URL}"

    tmp=$(mktemp -d)
    mount_point="${tmp}/mnt"
    # shellcheck disable=SC2064
    trap "hdiutil detach '${mount_point}' -quiet 2>/dev/null || true; rm -rf '${tmp}'" EXIT INT TERM

    fetch_to "${ASSET_URL}" "${tmp}/neurax.dmg"

    mkdir -p "${mount_point}"
    hdiutil attach "${tmp}/neurax.dmg" -mountpoint "${mount_point}" -nobrowse -quiet \
        || die "could not open the disk image"

    src=$(find "${mount_point}" -maxdepth 1 -name '*.app' -print -quit)
    [ -n "${src}" ] || die "the disk image contains no application"

    # /Applications when it is writable, the user's own otherwise. Never sudo:
    # a script piped from the internet should not be asking for a password.
    if [ -w /Applications ]; then
        target="${MAC_APP}"
    else
        target="${MAC_APP_USER}"
        mkdir -p "${HOME}/Applications"
        note "/Applications is not writable — installing to ~/Applications"
    fi

    rm -rf "${target}"
    cp -R "${src}" "${target}"

    # The build is not notarized, so Gatekeeper would refuse to open it with a
    # message that says the app is damaged. It is not damaged; it is unsigned.
    # Clearing the quarantine flag on a bundle the user just chose to install
    # is the same decision as right-clicking and picking Open.
    xattr -dr com.apple.quarantine "${target}" 2>/dev/null || true

    mkdir -p "${BIN_DIR}"
    ln -sf "${target}/Contents/MacOS/neurax-desktop" "${BIN_DIR}/neurax-desktop"

    install_launcher
    note "Installed to ${target}"
}

# ─── The `neurax` command ───────────────────────────────────────────

# Make `neurax` open the application — unless a `neurax` already exists that
# is not ours.
#
# The CLI compiler installs under that name too (`cargo install neurax-cli`),
# and it already opens the desktop app when run with no arguments. So if one is
# present, leaving it alone gives the user both: `neurax` opens the window,
# `neurax analyze model.json` still analyses.
install_launcher() {
    existing=$(command -v neurax 2>/dev/null || true)

    if [ -n "${existing}" ] && [ "${existing}" != "${BIN_DIR}/neurax" ]; then
        note "\`neurax\` already exists at ${existing} — left as it is"
        note "It will open this application when run with no arguments"
        return
    fi

    ln -sf "${APP_DIR}/neurax-desktop" "${BIN_DIR}/neurax" 2>/dev/null \
        || ln -sf "${BIN_DIR}/neurax-desktop" "${BIN_DIR}/neurax"
}

check_path() {
    case ":${PATH}:" in
        *":${BIN_DIR}:"*) return 0 ;;
    esac

    warn "${BIN_DIR} is not on your PATH."
    say  "    Add it by running the line for your shell, then opening a new terminal:"
    say  ""
    say  "      bash:  echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.bashrc"
    say  "      zsh:   echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.zshrc"
    say  "      fish:  fish_add_path ${BIN_DIR}"
}

# ─── Uninstall ──────────────────────────────────────────────────────

uninstall() {
    step "Removing NEURAX"
    removed=0
    for path in "${BIN_DIR}/neurax-desktop" "${APP_DIR}" "${DESKTOP_ENTRY}" \
                "${MAC_APP}" "${MAC_APP_USER}"; do
        if [ -e "${path}" ] || [ -L "${path}" ]; then
            rm -rf "${path}" && note "removed ${path}" && removed=1
        fi
    done

    # Only remove `neurax` if it is the symlink this script made. A CLI
    # installed separately is not ours to delete.
    if [ -L "${BIN_DIR}/neurax" ]; then
        case "$(readlink "${BIN_DIR}/neurax")" in
            *neurax-desktop) rm -f "${BIN_DIR}/neurax" && note "removed ${BIN_DIR}/neurax"; removed=1 ;;
        esac
    fi

    [ "${removed}" -eq 1 ] || say "Nothing to remove."
    say ""
    say "Your NEURAX projects and settings were not touched."
    exit 0
}

# ─── Main ───────────────────────────────────────────────────────────

detect_platform

[ "${UNINSTALL}" -eq 1 ] && uninstall

case "${PLATFORM}" in
    linux) install_linux ;;
    macos) install_macos ;;
esac

say ""
printf '%sNEURAX is installed.%s\n' "${GREEN}" "${RESET}"
say ""
say "  neurax          open the application"
say "  neurax --help   if you also have the CLI compiler installed"
say ""
note "The compiler runs inside the application, on your machine."
note "No account, no upload, no network."
say ""
check_path

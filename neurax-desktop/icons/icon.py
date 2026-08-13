#!/usr/bin/env python3
"""Assemble `icon.icns` and `icon.ico` from `master-1024.png`.

Split from `generate.mjs` because `.icns` and `.ico` are multi-resolution
containers rather than images, and Pillow writes both without needing the
platform tools (`iconutil` is macOS-only, and ImageMagick is not a dependency
here).

Run `node generate.mjs` first to produce the master.

Usage:  python3 icon.py
"""

from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
MASTER = HERE / "master-1024.png"

# Windows picks the nearest size at each UI scale; supplying only 256 makes the
# taskbar downscale it and the mark's thin strokes disappear.
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def main() -> None:
    if not MASTER.exists():
        raise SystemExit(f"{MASTER.name} is missing — run `node generate.mjs` first")

    master = Image.open(MASTER).convert("RGBA")

    master.resize((256, 256), Image.Resampling.LANCZOS).save(
        HERE / "icon.ico", sizes=[(s, s) for s in ICO_SIZES]
    )
    print(f"wrote icon.ico ({len(ICO_SIZES)} sizes)")

    master.save(HERE / "icon.icns", format="ICNS")
    print("wrote icon.icns")


if __name__ == "__main__":
    main()

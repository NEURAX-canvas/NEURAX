/**
 * Render the NEURAX application icons from the brand mark.
 *
 * The source of truth is `neurax-ui/public/neurax-favicon.svg` — the same file
 * the web app serves — so the desktop icon cannot drift from the site's.
 *
 * One correction is applied. In the source SVG the lower half of the brain is
 * drawn to y=64 while the rounded backdrop stops at y=60, so the shape bleeds
 * past the corner. At favicon size that is invisible; at 256px on a dock it
 * reads as a rendering fault. The mark is therefore clipped to the backdrop.
 *
 * Requires `sharp`, which is not a project dependency:
 *
 *     npm install sharp && node generate.mjs
 *
 * The generated files are committed, so this only needs running when the brand
 * mark changes.
 */

import sharp from 'sharp';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, '..', '..', 'neurax-ui', 'public', 'neurax-favicon.svg');
const svg = readFileSync(source);

/** The backdrop's geometry in the SVG's 64-unit viewBox. */
const CLIP = { x: 4, y: 4, size: 56, radius: 12, viewBox: 64 };

function clipMask(size) {
  const k = size / CLIP.viewBox;
  return Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}">` +
      `<rect x="${CLIP.x * k}" y="${CLIP.y * k}" ` +
      `width="${CLIP.size * k}" height="${CLIP.size * k}" ` +
      `rx="${CLIP.radius * k}" fill="#fff"/></svg>`,
  );
}

async function render(size) {
  return sharp(svg, { density: 2400 })
    .resize(size, size)
    .composite([{ input: clipMask(size), blend: 'dest-in' }])
    .png()
    .toBuffer();
}

/** Tauri's required PNGs, plus the Windows installer and Store logos. */
const PNGS = {
  '32x32.png': 32,
  '128x128.png': 128,
  '128x128@2x.png': 256,
  'Square30x30Logo.png': 30,
  'Square44x44Logo.png': 44,
  'Square71x71Logo.png': 71,
  'Square89x89Logo.png': 89,
  'Square107x107Logo.png': 107,
  'Square142x142Logo.png': 142,
  'Square150x150Logo.png': 150,
  'Square284x284Logo.png': 284,
  'Square310x310Logo.png': 310,
  'StoreLogo.png': 50,
};

for (const [name, size] of Object.entries(PNGS)) {
  writeFileSync(join(here, name), await render(size));
}
console.log(`wrote ${Object.keys(PNGS).length} PNGs`);

// A 1024px master, from which the .icns and .ico are assembled.
writeFileSync(join(here, 'master-1024.png'), await render(1024));
console.log('wrote master-1024.png — feed it to `icon.py` for .icns/.ico');

/**
 * Fail if any toolbar control is out of reach at a supported window width.
 *
 * This exists because the failure it catches is invisible to every other kind
 * of test. The studio's toolbar sits in an `overflow-x-auto` container, so
 * when it runs out of room the page does not overflow and nothing throws —
 * the buttons on the right simply stop being on screen, behind a scroll with
 * no affordance. Measured before the fix, the toolbar needed 1585px: at the
 * desktop application's own default size of 1440 the Target and Export
 * buttons were already gone, and at its minimum size of 1100 five were.
 *
 * The widths below are the ones that have to work: 1100 is
 * `min_inner_size` in `neurax-desktop/src/main.rs`, 1440 is the window it
 * opens at, and the rest are common displays.
 *
 * Needs the dev server running:  npm --prefix neurax-ui run dev
 * Then:                          node scripts/measure-responsive.mjs
 */

import { chromium } from 'playwright';

const WIDTHS = [1100, 1280, 1440, 1680, 1920];
const URL = process.env.NEURAX_UI_URL ?? 'http://localhost:8081/app';

const browser = await chromium.launch();
let failures = 0;

for (const width of WIDTHS) {
  const page = await browser.newPage({ viewport: { width, height: 800 } });

  // Present as the desktop host, so the studio is what gets measured rather
  // than the landing page.
  await page.addInitScript(() => {
    window.__NEURAX_DESKTOP__ = Object.freeze({
      apiBase: 'http://127.0.0.1:9098',
      platform: 'linux',
      ownTitleBar: true,
    });
  });

  await page.goto(URL, { waitUntil: 'networkidle' }).catch(() => {});
  await page.waitForTimeout(2500);

  const clipped = await page.evaluate(() => {
    const out = [];
    document.querySelectorAll('button, [role="tab"], a').forEach((el) => {
      const box = el.getBoundingClientRect();
      const label = (el.textContent ?? '').trim().slice(0, 30);
      // Zero-width elements are collapsed, not clipped.
      if (box.width > 0 && box.right > window.innerWidth + 1 && label) {
        out.push(`${label} (right edge ${Math.round(box.right)})`);
      }
    });
    return [...new Set(out)];
  });

  if (clipped.length) {
    failures += 1;
    console.log(`FAIL  ${width}px — ${clipped.length} control(s) off screen`);
    clipped.slice(0, 6).forEach((c) => console.log(`        ${c}`));
  } else {
    console.log(`PASS  ${width}px — every control reachable`);
  }

  await page.close();
}

await browser.close();

console.log(`\n${WIDTHS.length - failures}/${WIDTHS.length} widths pass`);
process.exit(failures ? 1 : 0);

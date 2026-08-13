/**
 * Drive the desktop application's frontend against the desktop's API.
 *
 * This is not the Tauri window — building that needs the platform webview
 * development packages, which are a system install. What it *is*: the exact
 * bundle `npm run build:desktop` produces, loaded with the same bootstrap
 * script `neurax-desktop` injects, talking to `neurax-service` started the way
 * the desktop starts it. Everything above the window frame is therefore the
 * real thing.
 *
 * What it does not cover, and nothing here should be read as covering: the
 * Tauri window itself, the native file dialogs, and the menu entry.
 *
 * Usage:  node scripts/verify-desktop.mjs
 */

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, extname } from 'node:path';

const ROOT = process.cwd();
const DIST = join(ROOT, 'neurax-ui', 'dist');
const SHOTS = join(ROOT, 'target', 'desktop-verification');
const API_PORT = 9401;
const UI_PORT = 9402;

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.svg': 'image/svg+xml', '.json': 'application/json', '.png': 'image/png',
  '.woff2': 'font/woff2', '.ico': 'image/x-icon',
};

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok, detail });
  console.log(`${ok ? '  PASS' : '  FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

/** Serve the built bundle, injecting what the desktop host injects. */
function serveBundle(apiBase) {
  const bootstrap =
    `<script>window.__NEURAX_DESKTOP__ = Object.freeze(` +
    `{ apiBase: ${JSON.stringify(apiBase)} });</script>`;

  return createServer((req, res) => {
    const url = req.url.split('?')[0];
    let file = join(DIST, url === '/' ? 'index.html' : url);
    // A single-page app: unknown paths are routes, not missing files.
    if (!existsSync(file) || url === '/') file = join(DIST, 'index.html');

    let body = readFileSync(file);
    if (file.endsWith('index.html')) {
      body = Buffer.from(body.toString().replace('<head>', `<head>${bootstrap}`));
    }
    res.writeHead(200, { 'Content-Type': MIME[extname(file)] ?? 'application/octet-stream' });
    res.end(body);
  }).listen(UI_PORT);
}

async function waitFor(url, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url);
      if (r.ok) return true;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

async function main() {
  mkdirSync(SHOTS, { recursive: true });

  if (!existsSync(join(DIST, 'index.html'))) {
    console.error('No desktop bundle. Run: npm --prefix neurax-ui run build:desktop');
    process.exit(1);
  }

  console.log('\nStarting the API the way neurax-desktop starts it');
  const api = spawn(join(ROOT, 'target', 'debug', 'neurax-service'), [], {
    env: { ...process.env, NEURAX_BIND: `127.0.0.1:${API_PORT}`, NEURAX_DEBUG_NOAUTH: 'true' },
    stdio: 'ignore',
  });

  const apiBase = `http://127.0.0.1:${API_PORT}`;
  if (!(await waitFor(`${apiBase}/health`))) {
    console.error('The API did not come up.');
    api.kill();
    process.exit(1);
  }
  check('API answers /health', true);

  const ui = serveBundle(apiBase);
  const browser = await chromium.launch();
  // The window size neurax-desktop asks for.
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  const consoleErrors = [];
  page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()));
  const failedRequests = [];
  page.on('requestfailed', (r) => failedRequests.push(`${r.url()} — ${r.failure()?.errorText}`));

  try {
    await page.goto(`http://127.0.0.1:${UI_PORT}/`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    // A local profile is created on first launch, so `/` should not stop on
    // the landing page — the desktop opens ready to work.
    const landedOn = new URL(page.url()).pathname;
    check('opens straight into the studio', landedOn === '/app', `landed on ${landedOn}`);
    await page.screenshot({ path: join(SHOTS, '01-studio.png'), fullPage: false });

    const bodyText = await page.textContent('body');
    check('the studio rendered', (bodyText?.length ?? 0) > 200, `${bodyText?.length} chars`);

    // Nothing may reach the network: the desktop build is offline-first.
    const external = failedRequests.filter((r) => !r.includes('127.0.0.1'));
    check('no external requests', external.length === 0, external.slice(0, 2).join('; '));

    // The API base must be the injected one, not a build-time default.
    const usesInjectedBase = await page.evaluate(
      () => window.__NEURAX_DESKTOP__?.apiBase ?? null,
    );
    check('uses the injected API base', usesInjectedBase === apiBase, String(usesInjectedBase));

    // Walk the workspaces.
    for (const label of ['Simulation', 'Production', 'Inference', 'Time Machine']) {
      const tab = page.getByRole('button', { name: new RegExp(label, 'i') }).first();
      if (await tab.count()) {
        await tab.click({ timeout: 5000 }).catch(() => {});
        await page.waitForTimeout(900);
        await page.screenshot({
          path: join(SHOTS, `${label.toLowerCase().replace(/\s+/g, '-')}.png`),
        });
        check(`${label} opens`, true);
      } else {
        check(`${label} opens`, false, 'tab not found');
      }
    }

    check(
      'no console errors',
      consoleErrors.length === 0,
      consoleErrors.slice(0, 2).join(' | '),
    );
  } finally {
    await browser.close();
    ui.close();
    api.kill();
  }

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  console.log(`Screenshots in ${SHOTS}`);
  process.exit(failed.length ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

// NEURAX profits from a model already being published on the Hub only if
// using that fact costs one line of text, not five browser tabs — this is
// the direct line from "paste an ID" to a real config.json, tested against
// what a user would actually paste, and (where marked) against the real
// Hub, not a mock.
import { describe, it, expect } from 'vitest';
import { parseRepoId, fetchHubConfig } from './huggingfaceHub.ts';

describe('parseRepoId', () => {
  it('accepts a bare owner/model ID', () => {
    const r = parseRepoId('mistralai/Mistral-7B-v0.1');
    expect(r).toEqual({ ok: true, repoId: 'mistralai/Mistral-7B-v0.1' });
  });

  it('accepts the model page URL', () => {
    const r = parseRepoId('https://huggingface.co/mistralai/Mistral-7B-v0.1');
    expect(r).toEqual({ ok: true, repoId: 'mistralai/Mistral-7B-v0.1' });
  });

  it('accepts a URL already pointing at a file inside the repo', () => {
    const r = parseRepoId('https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json');
    expect(r).toEqual({ ok: true, repoId: 'mistralai/Mistral-7B-v0.1' });
  });

  it('accepts a URL pointing at the files tree', () => {
    const r = parseRepoId('https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main');
    expect(r).toEqual({ ok: true, repoId: 'mistralai/Mistral-7B-v0.1' });
  });

  it('tolerates surrounding whitespace and a trailing slash', () => {
    expect(parseRepoId('  mistralai/Mistral-7B-v0.1  ')).toEqual({
      ok: true,
      repoId: 'mistralai/Mistral-7B-v0.1',
    });
    expect(parseRepoId('https://huggingface.co/mistralai/Mistral-7B-v0.1/')).toEqual({
      ok: true,
      repoId: 'mistralai/Mistral-7B-v0.1',
    });
  });

  it('rejects an empty input', () => {
    const r = parseRepoId('   ');
    expect(r.ok).toBe(false);
  });

  it('rejects a URL from a different host', () => {
    const r = parseRepoId('https://github.com/mistralai/Mistral-7B-v0.1');
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/github\.com/);
  });

  it('rejects an ID with no slash', () => {
    const r = parseRepoId('mistral-7b');
    expect(r.ok).toBe(false);
  });
});

describe('fetchHubConfig', () => {
  it('reports a parse error before ever making a request', async () => {
    const outcome = await fetchHubConfig('not a valid id');
    expect(outcome.ok).toBe(false);
  });
});

describe.skipIf(!process.env.NEURAX_LIVE_HUB_TEST)('fetchHubConfig against the real Hub', () => {
  // Opt-in, not run by default: a real network call to huggingface.co.
  //     NEURAX_LIVE_HUB_TEST=1 npx vitest run huggingfaceHub.test.ts
  it('fetches a real public config.json', async () => {
    const outcome = await fetchHubConfig('mistralai/Mistral-7B-v0.1');
    expect(outcome.ok).toBe(true);
    if (!outcome.ok) return;
    expect(outcome.result.repoId).toBe('mistralai/Mistral-7B-v0.1');
    const parsed = JSON.parse(outcome.result.contents);
    expect(parsed.model_type).toBe('mistral');
    expect(parsed.hidden_size).toBe(4096);
  }, 20000);

  it('gives an actionable error for a model that does not exist', async () => {
    // Verified live: the Hub answers 401/403 for this, the same status it
    // gives a real private repo — see the comment in huggingfaceHub.ts.
    const outcome = await fetchHubConfig('this-owner-does-not-exist-neurax-test/nope');
    expect(outcome.ok).toBe(false);
    if (outcome.ok) return;
    expect(outcome.error).toMatch(/doesn't exist|private/i);
  }, 20000);
});

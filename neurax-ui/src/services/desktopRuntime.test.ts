import { describe, it, expect, afterEach, vi } from 'vitest';
import { isDesktop, resolveApiBase, saveTextFile, openTextFile } from './desktopRuntime.ts';

function injectBridge(apiBase: unknown) {
  (window as unknown as Record<string, unknown>).__NEURAX_DESKTOP__ = { apiBase };
}

afterEach(() => {
  delete (window as unknown as Record<string, unknown>).__NEURAX_DESKTOP__;
  vi.restoreAllMocks();
});

describe('host detection', () => {
  it('reports a browser when nothing is injected', () => {
    expect(isDesktop()).toBe(false);
  });

  it('reports desktop when the bridge carries an API base', () => {
    injectBridge('http://127.0.0.1:41234');
    expect(isDesktop()).toBe(true);
  });

  it.each([
    ['an empty string', ''],
    ['a number', 4123],
    ['null', null],
    ['an object', {}],
  ])('falls back to the browser when apiBase is %s', (_label, value) => {
    injectBridge(value);
    expect(isDesktop()).toBe(false);
  });
});

describe('resolveApiBase', () => {
  it('keeps the build-time value in a browser', () => {
    expect(resolveApiBase('http://127.0.0.1:9098')).toBe('http://127.0.0.1:9098');
  });

  it('prefers the injected value on desktop', () => {
    injectBridge('http://127.0.0.1:41234');
    expect(resolveApiBase('http://127.0.0.1:9098')).toBe('http://127.0.0.1:41234');
  });

  it('strips trailing slashes so callers can concatenate paths', () => {
    expect(resolveApiBase('http://127.0.0.1:9098///')).toBe('http://127.0.0.1:9098');
    injectBridge('http://127.0.0.1:41234/');
    expect(resolveApiBase('http://x')).toBe('http://127.0.0.1:41234');
  });

  /**
   * A malformed injection must not silently produce `http://undefined/analyze`;
   * the browser value has to survive it.
   */
  it('ignores a malformed bridge rather than propagating it', () => {
    injectBridge(undefined);
    expect(resolveApiBase('http://127.0.0.1:9098')).toBe('http://127.0.0.1:9098');
  });
});

describe('saveTextFile in a browser', () => {
  it('downloads through an anchor and releases the object URL', async () => {
    const created = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:stub');
    const revoked = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const click = vi.fn();
    vi.spyOn(document, 'createElement').mockReturnValue({
      click,
      set href(_v: string) {},
      set download(_v: string) {},
    } as unknown as HTMLAnchorElement);

    const result = await saveTextFile('{"a":1}', 'model.neurax.json');

    expect(result).toEqual({ saved: true });
    expect(click).toHaveBeenCalledOnce();
    expect(created).toHaveBeenCalledOnce();
    // Leaking the URL keeps the blob alive for the life of the document, which
    // matters here because exports can be tens of megabytes.
    expect(revoked).toHaveBeenCalledWith('blob:stub');
  });

  it('reports no path, because a browser is never told one', async () => {
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:stub');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    vi.spyOn(document, 'createElement').mockReturnValue({
      click: vi.fn(),
      set href(_v: string) {},
      set download(_v: string) {},
    } as unknown as HTMLAnchorElement);

    expect((await saveTextFile('x', 'a.json')).path).toBeUndefined();
  });
});

describe('openTextFile in a browser', () => {
  it('resolves null when the picker yields no file', async () => {
    const input: Record<string, unknown> = { files: [], click: vi.fn() };
    vi.spyOn(document, 'createElement').mockImplementation(() => {
      queueMicrotask(() => (input.onchange as () => void)());
      return input as unknown as HTMLInputElement;
    });

    await expect(openTextFile(['json'])).resolves.toBeNull();
  });

  it('builds an accept list from the extensions', async () => {
    const input: Record<string, unknown> = { files: [], click: vi.fn() };
    vi.spyOn(document, 'createElement').mockImplementation(() => {
      queueMicrotask(() => (input.onchange as () => void)());
      return input as unknown as HTMLInputElement;
    });

    await openTextFile(['json', 'neurax']);
    expect(input.accept).toBe('.json,.neurax');
  });
});

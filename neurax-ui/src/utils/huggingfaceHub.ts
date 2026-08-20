/**
 * Fetching a `config.json` directly from the HuggingFace Hub.
 *
 * Before this, importing a model meant leaving NEURAX, finding the model on
 * the Hub, opening `config.json`, switching to raw view, copying the whole
 * file, and pasting it back — for every model, every time. NEURAX profits
 * from a model already being known and published on the Hub; that only
 * happens if using that fact costs one line of text, not five browser tabs.
 *
 * The Hub's raw-file endpoint (`huggingface.co/<repo>/raw/main/<path>`)
 * requires no authentication for a public repository and sends
 * `Access-Control-Allow-Origin` on its responses — verified directly against
 * the live endpoint, not assumed from documentation — so a plain browser
 * `fetch` reaches it from both the web app and the desktop build (whose CSP
 * already allows outbound `https:` connections).
 */

/** A parsed `owner/repo`, or the reason the input couldn't be read as one. */
export type ParsedRepoId =
  | { ok: true; repoId: string }
  | { ok: false; error: string };

/**
 * Read a model ID or a Hub URL down to `owner/repo`.
 *
 * Accepts what someone would actually paste: a bare ID
 * (`mistralai/Mistral-7B-v0.1`), the model's page URL, or a URL that already
 * points at a file inside it (`.../blob/main/config.json`,
 * `.../tree/main`) — the last of those because pointing at the exact file
 * you want is a reasonable thing to do even though this always re-derives
 * the path itself, to always fetch `config.json` specifically rather than
 * whatever file happened to be in the URL.
 */
export function parseRepoId(input: string): ParsedRepoId {
  const trimmed = input.trim();
  if (!trimmed) {
    return { ok: false, error: 'Enter a model ID (owner/model) or a Hub URL.' };
  }

  let path = trimmed;
  if (/^https?:\/\//i.test(trimmed)) {
    let url: URL;
    try {
      url = new URL(trimmed);
    } catch {
      return { ok: false, error: 'That does not look like a valid URL.' };
    }
    if (!/(^|\.)huggingface\.co$/i.test(url.hostname)) {
      return { ok: false, error: `"${url.hostname}" is not huggingface.co — paste the model's Hub URL, or just its ID (owner/model).` };
    }
    path = url.pathname;
  }

  // owner/repo, optionally followed by /resolve/main/..., /blob/main/...,
  // /tree/main, or a trailing slash — every shape a Hub URL's path takes.
  const match = path
    .replace(/^\/+/, '')
    .match(/^([^/]+)\/([^/]+)/);
  if (!match) {
    return {
      ok: false,
      error: 'Expected an ID shaped like "owner/model", e.g. mistralai/Mistral-7B-v0.1.',
    };
  }
  return { ok: true, repoId: `${match[1]}/${match[2]}` };
}

export interface HubFetchResult {
  repoId: string;
  contents: string;
}

/**
 * What the Hub knows about a repository beyond its `config.json` — who
 * published it, when, under what license, and how established it is.
 *
 * None of this changes a single number NEURAX computes: the architecture
 * import reads `config.json` alone, exactly as before. This is context for
 * the human deciding whether to trust what they just imported — a shape
 * that happens to parse is not evidence the repository is real, current, or
 * the one they meant to reference.
 */
export interface HubModelInfo {
  repoId: string;
  /** The organization or user that published it — not necessarily who wrote it. */
  author: string | null;
  license: string | null;
  downloads: number;
  likes: number;
  /** When the repository was created on the Hub. */
  createdAt: string | null;
  /** When any file in it — weights, config, README — was last pushed. */
  lastModified: string | null;
  gated: boolean;
  private: boolean;
}

/** A `license:<id>` tag, the fallback when `cardData.license` is absent. */
function licenseFromTags(tags: unknown): string | null {
  if (!Array.isArray(tags)) return null;
  const tag = tags.find((t): t is string => typeof t === 'string' && t.startsWith('license:'));
  return tag ? tag.slice('license:'.length) : null;
}

/**
 * Fetch what the Hub's model-info API knows about a repository.
 *
 * `huggingface.co/api/models/<repo>` — verified directly against the live
 * endpoint (not assumed from documentation): public, unauthenticated, and
 * sends `Access-Control-Allow-Origin` for the caller's own origin, the same
 * way the raw-file endpoint `fetchHubConfig` uses does.
 *
 * Best-effort and separate from `fetchHubConfig` on purpose: this call
 * failing (rate limit, a Hub outage, a repo whose info endpoint 404s for a
 * reason its raw-file endpoint didn't) must never block an import that only
 * needs the config. Callers get `null` rather than a thrown error for
 * exactly that reason — there is nothing actionable a user could do about
 * missing publish metadata, unlike a missing or malformed config.
 */
export async function fetchHubModelInfo(repoId: string): Promise<HubModelInfo | null> {
  let response: Response;
  try {
    response = await fetch(`https://huggingface.co/api/models/${repoId}`, {
      signal: AbortSignal.timeout(10_000),
    });
  } catch {
    return null;
  }
  if (!response.ok) return null;

  let data: Record<string, unknown>;
  try {
    data = await response.json();
  } catch {
    return null;
  }

  const cardData = data.cardData;
  const cardLicense =
    cardData && typeof cardData === 'object' && !Array.isArray(cardData)
      ? (cardData as Record<string, unknown>).license
      : undefined;

  return {
    repoId,
    author: typeof data.author === 'string' ? data.author : null,
    license: typeof cardLicense === 'string' ? cardLicense : licenseFromTags(data.tags),
    downloads: typeof data.downloads === 'number' ? data.downloads : 0,
    likes: typeof data.likes === 'number' ? data.likes : 0,
    createdAt: typeof data.createdAt === 'string' ? data.createdAt : null,
    lastModified: typeof data.lastModified === 'string' ? data.lastModified : null,
    gated: data.gated !== false && data.gated !== undefined,
    private: data.private === true,
  };
}

/**
 * Fetch a repo's `config.json` from the Hub.
 *
 * Tries `main` first, then `master` — a shrinking but real minority of
 * older repos never renamed their default branch. Errors are returned
 * rather than thrown, matching every other parser in this codebase, so the
 * dialog can show one beside the input instead of unmounting.
 */
export async function fetchHubConfig(input: string): Promise<
  | { ok: true; result: HubFetchResult }
  | { ok: false; error: string }
> {
  const parsed = parseRepoId(input);
  if (!parsed.ok) return { ok: false, error: parsed.error };
  const { repoId } = parsed;

  for (const branch of ['main', 'master']) {
    const url = `https://huggingface.co/${repoId}/raw/${branch}/config.json`;
    let response: Response;
    try {
      response = await fetch(url, { signal: AbortSignal.timeout(15_000) });
    } catch (err) {
      // A network failure on the *first* branch tried is worth surfacing
      // immediately rather than silently trying the second one and hiding
      // what actually went wrong (offline, DNS, a corporate proxy).
      return {
        ok: false,
        error: `Could not reach huggingface.co: ${err instanceof Error ? err.message : String(err)}. Check your connection.`,
      };
    }

    if (response.ok) {
      const contents = await response.text();
      return { ok: true, result: { repoId, contents } };
    }

    if (response.status === 401 || response.status === 403) {
      // Verified against the live Hub, not assumed: it answers with the same
      // status for a repository that doesn't exist and one that does but is
      // private or gated — apparently deliberate, so an unauthenticated
      // caller can't use the difference to detect a private repo's
      // existence. This can't tell the two apart, so it doesn't claim to.
      return {
        ok: false,
        error: `"${repoId}" isn't reachable — either it doesn't exist (check the ID is exact; case matters) or it's private/gated, which NEURAX can't fetch. Visit huggingface.co/${repoId} to check, or download its config.json yourself and use Open File instead.`,
      };
    }
    // 404 on `main`: fall through and try `master` before giving up.
  }

  return {
    ok: false,
    error: `No config.json found at "${repoId}" on either "main" or "master". Check the ID is exact (case matters) — visit huggingface.co/${repoId} to confirm it exists and is a model, not a dataset or Space.`,
  };
}

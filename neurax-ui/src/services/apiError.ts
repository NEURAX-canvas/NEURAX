/**
 * The API error type, on its own.
 *
 * Deliberately a module of its own rather than a class inside `neuraxApi.ts`.
 * Anything that wants to recognise a failed request needs only this type, but
 * importing it from the API module pulls in the whole client — and with it the
 * Supabase client, which throws at import time when `VITE_SUPABASE_URL` is
 * absent. That is fine for the application, which always has it configured, and
 * wrong for a unit test of error *wording*, which needs no network layer at all
 * and failed in CI for the lack of a variable it never uses.
 *
 * `neuraxApi.ts` re-exports this, so existing imports are unaffected.
 */

/** A request that reached the server and came back with a failure status. */
export class NeuraxApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    /** The response body, kept whole: it carries the compiler's own message. */
    public body?: unknown,
  ) {
    // Include the body when there is one. `String(err)` is what ends up in a
    // log or a toast at the edges, and "Neurax API 400: Bad Request" names
    // nothing the reader can act on.
    const detail =
      typeof body === 'string' && body.trim() ? ` — ${body.trim().slice(0, 200)}` : '';
    super(`Neurax API ${status}: ${statusText}${detail}`);
    this.name = 'NeuraxApiError';
  }
}

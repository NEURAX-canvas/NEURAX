/**
 * The compiler's words must reach the user.
 *
 * Every message asserted here was captured from a running `neurax-service`, by
 * posting a broken topology and recording exactly what came back. That matters:
 * the previous handler failed because it assumed the 400 body was a JSON
 * diagnostics array, and it is plain text. Testing against invented strings
 * would have reproduced that mistake.
 *
 * The property that matters most is the last one: whatever arrives, something
 * useful is shown. There is no input to `explainAnalysisFailure` that produces
 * an empty explanation or a bare "something went wrong".
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { explainAnalysisFailure, failureAsWarnings } from './compilerErrors';
import { NeuraxApiError } from './neuraxApi';

/** Responses captured from a live neurax-service on a broken topology. */
const REAL = {
  noGpu: 'Schema validation error: At least one GPU must be specified',
  noLayers: 'Schema validation error: Model must have at least one layer',
  missingSchema: 'JSON parsing error: missing field `schema_version` at line 1 column 2',
  missingTopology: 'Json deserialize error: missing field `topology` at line 1 column 10',
};

const bad = (body: string, status = 400) =>
  new NeuraxApiError(status, 'Bad Request', body);

afterEach(() => {
  delete (window as { __NEURAX_DESKTOP__?: unknown }).__NEURAX_DESKTOP__;
  vi.restoreAllMocks();
});

describe('explaining a failed analysis', () => {
  describe('turns the compiler\'s real messages into something actionable', () => {
    it('points at the Target dialog when no GPU is selected', () => {
      const failure = explainAnalysisFailure(bad(REAL.noGpu));
      expect(failure.title).toBe('No target hardware');
      expect(failure.hint).toMatch(/Target/);
      expect(failure.internal).toBe(false);
    });

    it('points at the palette when the canvas is empty', () => {
      const failure = explainAnalysisFailure(bad(REAL.noLayers));
      expect(failure.title).toBe('Nothing to analyse');
      expect(failure.hint).toMatch(/palette|template/i);
    });

    it('blames NEURAX, not the design, when the IR is malformed', () => {
      // A missing `schema_version` means NEURAX produced a bad document. The
      // user cannot fix that by editing blocks, and sending them to look is
      // worse than saying nothing.
      for (const body of [REAL.missingSchema, REAL.missingTopology]) {
        const failure = explainAnalysisFailure(bad(body));
        expect(failure.internal, body).toBe(true);
        expect(failure.hint, body).toMatch(/defect in NEURAX/i);
      }
    });

    it('strips the compiler\'s internal prefix from what it shows', () => {
      const failure = explainAnalysisFailure(bad('Schema validation error: something odd happened'));
      expect(failure.detail).toBe('something odd happened');
      expect(failure.detail).not.toMatch(/schema validation error/i);
    });
  });

  describe('never discards what the compiler said', () => {
    it('shows an unrecognised message verbatim rather than a generic one', () => {
      const novel = 'Tensor rank 7 exceeds the supported maximum of 6 on layer conv_3';
      const failure = explainAnalysisFailure(bad(novel));
      expect(failure.detail).toBe(novel);
      expect(failure.detail).not.toMatch(/topology|went wrong/i);
    });

    it('keeps the raw text even when it recognised the message', () => {
      expect(explainAnalysisFailure(bad(REAL.noGpu)).raw).toContain('At least one GPU');
    });

    it('does not throw on a body that only looks like JSON', () => {
      const failure = explainAnalysisFailure(bad('{ this is not json after all'));
      expect(failure.detail.length).toBeGreaterThan(0);
      expect(failure.detail).toContain('not json');
    });
  });

  describe('prefers structured diagnostics when the compiler sends them', () => {
    const withDiagnostics = JSON.stringify({
      report: {
        diagnostics: [
          {
            category: 'shape',
            severity: 'error',
            code: 'E_SHAPE',
            message: 'Input width 512 does not match output width 768',
            layer_id: 'n7',
            suggestion: 'Set the FFN width to match the model width.',
          },
          { category: 'precision', severity: 'warning', message: 'fp8 may be unstable here' },
        ],
      },
    });

    it('reports the error count and leads with the first error', () => {
      const failure = explainAnalysisFailure(bad(withDiagnostics));
      expect(failure.title).toMatch(/1 error/);
      expect(failure.detail).toMatch(/does not match/);
      expect(failure.diagnostics).toHaveLength(2);
    });

    it('carries the compiler\'s own suggestion through as the hint', () => {
      expect(explainAnalysisFailure(bad(withDiagnostics)).hint).toMatch(/Set the FFN width/);
    });

    it('counts only blocking diagnostics as errors', () => {
      const warningsOnly = JSON.stringify({
        report: { diagnostics: [{ severity: 'warning', message: 'expert load is uneven' }] },
      });
      const failure = explainAnalysisFailure(bad(warningsOnly));
      expect(failure.title).not.toMatch(/failed/i);
      expect(failure.title).toMatch(/1 issue/);
    });

    it('names the block a diagnostic belongs to in the issues panel', () => {
      const rows = failureAsWarnings(explainAnalysisFailure(bad(withDiagnostics)));
      expect(rows[0].message).toContain('block n7');
      expect(rows[0].type).toBe('error');
      expect(rows[1].type).toBe('warning');
    });
  });

  describe('handles the status codes that mean something specific', () => {
    it('asks the user to sign in on 401', () => {
      expect(explainAnalysisFailure(new NeuraxApiError(401, 'Unauthorized')).title).toMatch(/sign in/i);
    });

    it('distinguishes an unverified email from a plan limit on 403', () => {
      const verify = explainAnalysisFailure(
        new NeuraxApiError(403, 'Forbidden', 'please verify your email address'),
      );
      expect(verify.title).toMatch(/verify/i);

      const plan = explainAnalysisFailure(new NeuraxApiError(403, 'Forbidden', 'plan limit reached'));
      expect(plan.title).toMatch(/plan/i);
    });

    it('says to wait on 429', () => {
      expect(explainAnalysisFailure(new NeuraxApiError(429, 'Too Many Requests')).hint).toMatch(/wait/i);
    });

    it('treats a 5xx as a fault in NEURAX rather than in the design', () => {
      const failure = explainAnalysisFailure(new NeuraxApiError(500, 'Internal Server Error'));
      expect(failure.internal).toBe(true);
      expect(failure.hint).toMatch(/bug report/i);
    });
  });

  describe('adapts its wording to where it is running', () => {
    it('does not tell a desktop user to check their connection', () => {
      // In the desktop build the compiler is a thread in this process; "check
      // your connection" and "the server" both send the reader nowhere.
      (window as { __NEURAX_DESKTOP__?: unknown }).__NEURAX_DESKTOP__ = {
        apiBase: 'http://127.0.0.1:41234',
      };
      const failure = explainAnalysisFailure(new TypeError('Failed to fetch'));
      expect(failure.detail).toMatch(/inside NEURAX/i);
      expect(failure.hint).toMatch(/Restarting NEURAX/i);
      expect(`${failure.detail} ${failure.hint}`).not.toMatch(/your connection/i);
    });

    it('does tell a browser user to check their connection', () => {
      const failure = explainAnalysisFailure(new TypeError('Failed to fetch'));
      expect(failure.hint).toMatch(/connection/i);
      expect(failure.internal).toBe(false);
    });
  });

  describe('always produces something worth showing', () => {
    const anything: unknown[] = [
      new TypeError('Failed to fetch'),
      new NeuraxApiError(400, 'Bad Request', ''),
      new NeuraxApiError(404, 'Not Found'),
      new NeuraxApiError(500, 'Internal Server Error', 'thread panicked'),
      new NeuraxApiError(503, 'Service Unavailable'),
      bad(REAL.noGpu),
      bad(REAL.noLayers),
      bad(REAL.missingSchema),
      bad('{}'),
      bad('[]'),
      'a bare string',
      null,
      undefined,
      42,
    ];

    it('gives every failure a title and a detail', () => {
      for (const thrown of anything) {
        const failure = explainAnalysisFailure(thrown);
        expect(failure.title.length, String(thrown)).toBeGreaterThan(0);
        expect(failure.detail.length, String(thrown)).toBeGreaterThan(0);
      }
    });

    it('never shows the reader an empty issues panel while claiming a failure', () => {
      for (const thrown of anything) {
        const rows = failureAsWarnings(explainAnalysisFailure(thrown));
        expect(rows.length, String(thrown)).toBeGreaterThan(0);
        expect(rows[0].message.length).toBeGreaterThan(0);
      }
    });

    it('never says "backend", which names nothing the user can see', () => {
      for (const thrown of anything) {
        const failure = explainAnalysisFailure(thrown);
        const text = `${failure.title} ${failure.detail} ${failure.hint ?? ''}`;
        expect(text, String(thrown)).not.toMatch(/backend/i);
      }
    });
  });
});

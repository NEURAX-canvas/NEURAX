/**
 * Serialise a JSON-safe value (the compiled NEURAX IR, in practice) to TOML.
 *
 * Round-tripped through JSON first so the TOML export always describes
 * exactly the same object the JSON export does — whatever `JSON.stringify`
 * already normalises (dropped `undefined`s, functions, `NaN`/`Infinity`
 * collapsed to `null`) is what gets serialised to TOML too, rather than a
 * second, potentially-diverging pass over the source object directly.
 * `smol-toml` drops `null` fields rather than erroring on them — TOML has
 * no `null`, and a dropped key describes the same "nothing here" as JSON's
 * `null` does for every consumer that reads either format back. An array of
 * layer objects renders as TOML's own idiom for that shape — `[[layers]]`,
 * an array of tables, one block per layer — rather than a JSON-style inline
 * array of objects, which is what makes a large export readable and
 * diffable as TOML rather than JSON wearing a different file extension.
 */
import { stringify } from 'smol-toml';

export function toToml(value: unknown): string {
  const jsonSafe = JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
  return stringify(jsonSafe);
}

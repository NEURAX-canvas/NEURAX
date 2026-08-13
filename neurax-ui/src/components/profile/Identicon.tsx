/**
 * Deterministic identicons, in the style GitHub generates for accounts without
 * a picture.
 *
 * The pattern and the colour both come from a hash of the seed, so the same
 * seed always draws the same avatar — which is what makes an identicon useful
 * as an identity: a person recognises their own without having chosen it, and
 * two people never collide by accident on a small set of stock images.
 *
 * Everything is drawn as inline SVG. There is no image to fetch, nothing to
 * cache, and the avatar renders identically offline and in tests.
 */

/** Grid is 5 wide; only the left three columns are drawn, the rest mirrors. */
const GRID = 5;
const HALF = Math.ceil(GRID / 2);

/**
 * FNV-1a, 32-bit.
 *
 * Chosen for being short, dependency-free and well spread over short strings —
 * the avalanche matters here because neighbouring seeds ("user1", "user2")
 * should not produce neighbouring patterns.
 */
function hashSeed(seed: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) {
    hash ^= seed.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  // Final avalanche. FNV-1a alone leaves the low bits weakly mixed, and this
  // function is asked for one bit at a time: without the finaliser, twelve
  // catalogue seeds produced only ten distinct patterns.
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x846ca68b) >>> 0;
  hash ^= hash >>> 16;
  return hash >>> 0;
}

/**
 * Which cells are filled, as a flat GRID x GRID array.
 *
 * Bits of the hash decide the left half; the right half mirrors it, which is
 * what gives an identicon its face-like symmetry.
 */
export function identiconCells(seed: string): boolean[] {
  const cells: boolean[] = new Array(GRID * GRID).fill(false);

  for (let column = 0; column < HALF; column++) {
    for (let row = 0; row < GRID; row++) {
      // One well-mixed bit per cell of the left half. Each cell gets its own
      // hash so the fifteen decisions stay independent of one another.
      const bit = hashSeed(`${seed}:${column}:${row}`) & 1;
      if (!bit) continue;
      cells[row * GRID + column] = true;
      cells[row * GRID + (GRID - 1 - column)] = true;
    }
  }

  // A blank or nearly blank grid reads as a rendering failure rather than an
  // avatar, so guarantee the centre column when the pattern came out too sparse.
  if (cells.filter(Boolean).length < 5) {
    for (let row = 1; row < GRID - 1; row++) {
      cells[row * GRID + Math.floor(GRID / 2)] = true;
    }
  }

  return cells;
}

/** Foreground colour for a seed: a fixed saturation and lightness, hashed hue. */
export function identiconColor(seed: string): string {
  const hue = hashSeed(`${seed}:hue`) % 360;
  // Mid lightness and strong-but-not-neon saturation keep every generated hue
  // legible against both the light and the dark chart surfaces.
  return `hsl(${hue} 62% 48%)`;
}

export interface IdenticonProps {
  /** Anything stable that identifies the user: id, email, username. */
  seed: string;
  /** Rendered size in pixels. */
  size?: number;
  /** Rounded like a GitHub avatar, or square. */
  rounded?: boolean;
  className?: string;
  /** Overrides the accessible name; defaults to describing it as generated. */
  title?: string;
}

export function Identicon({
  seed,
  size = 40,
  rounded = true,
  className,
  title,
}: IdenticonProps) {
  const cells = identiconCells(seed);
  const color = identiconColor(seed);

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${GRID} ${GRID}`}
      className={className}
      style={{ borderRadius: rounded ? '50%' : size * 0.15, display: 'block' }}
      role="img"
      aria-label={title ?? `Generated avatar for ${seed}`}
      shapeRendering="crispEdges"
    >
      {/* The plate. `currentColor` lets the surrounding surface set it, so the
          avatar sits correctly in both themes without a second palette. */}
      <rect width={GRID} height={GRID} fill="currentColor" opacity={0.08} />
      {cells.map((filled, index) =>
        filled ? (
          <rect
            key={index}
            x={index % GRID}
            y={Math.floor(index / GRID)}
            width={1}
            height={1}
            fill={color}
          />
        ) : null,
      )}
    </svg>
  );
}

// Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices minute by minute, 240 minutes ahead.
//
// What it does, per market, from the case's own price history alone (no network, deterministic):
//   1. estimate the market's per-minute move scale robustly (median absolute deviation of one-minute
//      differences, plus their standard deviation, so a single jump does not set the scale);
//   2. measure the market's own multi-horizon displacement quantiles: for horizons 1, 10, 30, 60, 120 and
//      240 minutes, the empirical 5/25/50/75/95% quantiles of overlapping price changes over the last two
//      days of history. Prices move in steps and jump now and then, and these quantiles carry that shape;
//   3. blend each empirical quantile with the random-walk quantile z * sigma * sqrt(h), the empirical one
//      weighted by its effective sample size (a 240-minute change has few independent observations, so it
//      leans on the parametric shape), and nudge the two outer quantiles outward, which pinball loss makes
//      cheap: a 5% quantile that sits too low costs 0.05 per point, one that sits too high costs 0.95;
//   4. interpolate those quantile offsets across all 240 horizon rows in sqrt(h) and add them to the price
//      at the checkpoint, clamped to [0, 1] and ordered.
//
// Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, matching QUANTILES. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** The grid of horizons whose displacement quantiles are measured directly. */
const GRID = [1, 10, 30, 60, 120, 240];
/** One-minute differences used for the scale: the last two days. */
const SCALE_WINDOW = 2880;
/** Effective-sample size at which the empirical and parametric quantiles weigh the same. */
const PRIOR_EFFECTIVE = 12;
/** Below this, a market counts as still and the spread falls back to a small floor. */
const SIGMA_FLOOR = 0.0006;
/** The outer quantiles are widened by this much, which pinball loss prices cheaply. */
const TAIL_WIDEN = Number(process.env.MAKEFUN_TAIL ?? 1.26);
/** Global spread calibration, measured on the practice cases. */
const SPREAD_SCALE = Number(process.env.MAKEFUN_SCALE ?? 0.11);

interface CaseMarket {
  id: string;
  name?: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);

/** The q-quantile of `sorted` (ascending, non-empty), interpolating between neighbours. */
function quantileOf(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0] as number;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const a = sorted[lo] as number;
  const b = sorted[hi] as number;
  return a + (b - a) * (pos - lo);
}

/** A robust per-minute move scale from a market's one-minute differences. */
function minuteScale(prices: number[]): number {
  const n = prices.length;
  const from = Math.max(1, n - 1 - SCALE_WINDOW);
  const diffs: number[] = [];
  for (let t = from; t < n; t++) diffs.push((prices[t] as number) - (prices[t - 1] as number));
  if (diffs.length < 5) return SIGMA_FLOOR;
  const sorted = [...diffs].sort((a, b) => a - b);
  const median = quantileOf(sorted, 0.5);
  const absDev = diffs.map((d) => Math.abs(d - median)).sort((a, b) => a - b);
  const mad = quantileOf(absDev, 0.5) * 1.4826;
  let sum = 0;
  let sumSq = 0;
  for (const d of diffs) {
    sum += d;
    sumSq += d * d;
  }
  const mean = sum / diffs.length;
  const std = Math.sqrt(Math.max(0, sumSq / diffs.length - mean * mean));
  return Math.max(SIGMA_FLOOR, mad, std * 0.8);
}

interface HorizonQuantiles {
  /** sqrt(h), the interpolation axis. */
  axis: number;
  /** The five displacement quantiles at that horizon. */
  offsets: number[];
}

/** The five displacement quantiles at one horizon, empirical blended with the random walk. */
function horizonQuantiles(prices: number[], h: number, sigma: number): HorizonQuantiles {
  const n = prices.length;
  const from = Math.max(0, n - 1 - SCALE_WINDOW);
  const diffs: number[] = [];
  for (let t = from; t + h <= n - 1; t++) diffs.push((prices[t + h] as number) - (prices[t] as number));
  const parametric = Z.map((z) => z * sigma * Math.sqrt(h));
  if (diffs.length === 0) return { axis: Math.sqrt(h), offsets: parametric };
  diffs.sort((a, b) => a - b);
  // Overlapping windows overlap, so the count of independent changes is about diffs/h.
  const effective = diffs.length / Math.max(1, h);
  const wParametric = clamp(PRIOR_EFFECTIVE / (PRIOR_EFFECTIVE + effective), 0.05, 0.9);
  const offsets = QUANTILES.map((q, j) => {
    const empirical = quantileOf(diffs, q);
    return (1 - wParametric) * empirical + wParametric * (parametric[j] as number);
  });
  offsets[0] = Math.min(offsets[0] as number, 0) * TAIL_WIDEN;
  offsets[4] = Math.max(offsets[4] as number, 0) * TAIL_WIDEN;
  if (process.env.MAKEFUN_NODRIFT !== '0') {
    // Recentre on the checkpoint's own price: a 4-hour move's direction is not predictable from history,
    // so the median is persistence and only the shape around it is kept.
    const centre = offsets[2] as number;
    for (let j = 0; j < offsets.length; j++) offsets[j] = (offsets[j] as number) - centre;
    offsets[0] = Math.min(offsets[0] as number, 0) * TAIL_WIDEN;
    offsets[4] = Math.max(offsets[4] as number, 0) * TAIL_WIDEN;
  }
  return { axis: Math.sqrt(h), offsets };
}

/** The forecast of one market: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const prices = m.prices;
  const x = prices[prices.length - 1] as number;
  const sigma = minuteScale(prices);
  const grid = GRID.filter((h) => h <= horizon).map((h) => horizonQuantiles(prices, h, sigma));
  // A horizon past the grid's reach (only when horizon > 240) holds the last measured spread.
  if (grid.length === 0) grid.push(horizonQuantiles(prices, 1, sigma));
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const axis = Math.sqrt(h);
    let lo = 0;
    while (lo + 1 < grid.length && (grid[lo + 1] as HorizonQuantiles).axis <= axis) lo++;
    const a = grid[lo] as HorizonQuantiles;
    const b = (grid[lo + 1] ?? a) as HorizonQuantiles;
    const span = b.axis - a.axis;
    const f = span > 0 ? clamp((axis - a.axis) / span, 0, 1) : 0;
    const row = QUANTILES.map((_, j) => {
      const offset = ((a.offsets[j] as number) + ((b.offsets[j] as number) - (a.offsets[j] as number)) * f) * SPREAD_SCALE;
      return price6(x + offset);
    });
    row.sort((p, q) => p - q);
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (
    c.quantiles.length !== QUANTILES.length ||
    c.quantiles.some((t, j) => t !== QUANTILES[j])
  ) {
    throw new Error(
      `${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`,
    );
  }
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}

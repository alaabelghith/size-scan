/**
 * estimator.js — Size Scan measurement maths.
 *
 * Pure functions, no browser APIs, so they run (and are tested) in Node too.
 * Same method as python/size_estimator.py, the version that runs in the
 * Fashion Mirror kiosk:
 *
 *   1. Scale. The user's height and the eye-to-ankle distance in the image give
 *      centimetres per pixel (eyes at ~93.6 % of stature, ankles at ~3.9 %). With a
 *      fixed camera, predictHeight() can also estimate the height itself.
 *   2. Widths. On the front view, the silhouette is measured over three bands between
 *      the shoulder and hip landmarks: chest (median), natural waist (narrowest row),
 *      hips (30th percentile). Arms and hands touching the body are cut off.
 *   3. Depths. The same three lines are measured on the side view.
 *   4. Circumference. Each cross-section is an ellipse (width x depth, Ramanujan) times
 *      a shape factor, then a calibration factor; implausible depths / values are clamped.
 *   5. Size. Each circumference is matched against a size chart; a value near a
 *      boundary also returns the neighbouring size.
 *
 * Single frames are noisy, so the app collects every stable frame of a pose and
 * this module takes the median.
 */

// MediaPipe Pose landmark indices (33-point model)
export const LM = {
  NOSE: 0, LEFT_EYE: 2, RIGHT_EYE: 5,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14, LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
};

export const VIS_MIN = 0.55;            // landmark visibility needed to trust a point
export const EYE_HEIGHT_FRAC = 0.936;   // eye height / stature (anthropometric average)
export const ANKLE_HEIGHT_FRAC = 0.039; // ankle (malleolus) height / stature: the ankle point is NOT the floor
export const CHEST_FRAC = 0.18;         // chest line, as a fraction of the shoulder -> hip span
export const WAIST_FRAC = 0.62;         // waist line
export const MIN_MASK_ROW_PIXELS = 8;   // fewer foreground pixels than this = unreliable row
export const UPPER_ARM_HALF_CM = 4.5;   // half the thickness of an upper arm / forearm, used where an arm
export const FOREARM_HALF_CM = 3.5;     // touches the body in the front view and has to be cut off
export const HAND_FRAC = 0.8;           // the hand hangs ~0.8 forearm below the wrist point
export const MAX_GAP_FRAC = 0.004;      // background gap (fraction of image width) still counted as body
export const ARM_GAP_CM = 10;           // front pose: wrists at least this far outside the hips
// Cross-sections are rounder-rectangular than an ellipse: shape factors on top of Ramanujan.
export const SHAPE_FACTOR = { chest: 1.06, waist: 1.02, hip: 1.05 };
// Plausible side-depth / front-width ratios; outside = bad side view (arm in front, half turned) -> clamped.
export const DEPTH_RATIO = { chest: [0.58, 0.85], waist: [0.60, 0.95], hip: [0.62, 0.88] };
export const DEFAULT_DEPTH_RATIO = { chest: 0.70, waist: 0.75, hip: 0.72 }; // when no side view
// Plausible circumference / stature for adults; outside -> clamped and confidence low.
export const STATURE_BOUNDS = { chest: [0.45, 0.78], waist: [0.36, 0.72], hip: [0.47, 0.80] };
// Correction factors from test scans (waist and hips read high). Replace with your own tape-measure calibration.
export const DEFAULT_CALIBRATION = { chest: 1.0, waist: 0.88, hip: 0.9 };
// Measuring bands, as fractions of the shoulder (0) -> hip (1) span.
const BANDS = { chest: [0.12, 0.26, 9], waist: [0.45, 0.80, 13], hip: [0.92, 1.08, 9] };

/**
 * Size charts (cm, body measurements). Tops by chest, bottoms by waist (letter + French
 * trouser number), dresses by the larger of chest and hips. Men's tops follow Primark's chart;
 * trouser numbers are the French system (number ~ waist / 2, Celio / Kiabi), capped at 48 like
 * mass retail in France and Tunisia ("48+" above). Replace with your brand's charts.
 */
export const SIZE_ORDER = ['S', 'M', 'L', 'XL', 'XXL'];
export const SIZE_CHARTS = {
  women: {
    tops: { S: [82, 90], M: [90, 98], L: [98, 106], XL: [106, 116], XXL: [116, 128] },
    bottoms: {
      S: { waist: [66, 74], hip: [88, 96] }, M: { waist: [74, 82], hip: [96, 104] },
      L: { waist: [82, 90], hip: [104, 112] }, XL: { waist: [90, 100], hip: [112, 122] },
      XXL: { waist: [100, 114], hip: [122, 138] },
    },
    'one-pieces': {
      S: { chest: [82, 90], hip: [88, 96] }, M: { chest: [90, 98], hip: [96, 104] },
      L: { chest: [98, 106], hip: [104, 112] }, XL: { chest: [106, 116], hip: [112, 122] },
      XXL: { chest: [116, 128], hip: [122, 138] },
    },
  },
  men: {
    tops: { S: [90, 97], M: [97, 103], L: [103, 109], XL: [109, 119], XXL: [119, 128] },
    bottoms: {
      S: { waist: [74, 82], hip: [90, 98] }, M: { waist: [82, 90], hip: [98, 106] },
      L: { waist: [90, 98], hip: [106, 114] }, XL: { waist: [98, 108], hip: [114, 124] },
      XXL: { waist: [108, 120], hip: [124, 138] },
    },
    'one-pieces': {
      S: { chest: [90, 97], hip: [90, 98] }, M: { chest: [97, 103], hip: [98, 106] },
      L: { chest: [103, 109], hip: [106, 114] }, XL: { chest: [109, 119], hip: [114, 124] },
      XXL: { chest: [119, 128], hip: [124, 138] },
    },
  },
};
export const TROUSER_FR = {
  women: { 32: [60, 64], 34: [64, 68], 36: [68, 72], 38: [72, 76], 40: [76, 80], 42: [80, 84], 44: [84, 88], 46: [88, 92], 48: [92, 98] },
  men: { 36: [70, 74], 38: [74, 78], 40: [78, 82], 42: [82, 86], 44: [86, 90], 46: [90, 94], 48: [94, 98], 50: [98, 102], 52: [102, 106], 54: [106, 112] },
};
export const RETAIL_MAX = 48;

// ── geometry ────────────────────────────────────────────────────────────────

/** Perimeter of an ellipse with the given full width and depth (Ramanujan II). */
export function ellipseCircumference(width, depth) {
  const a = width / 2, b = depth / 2;
  if (a <= 0 || b <= 0) return 0;
  const h = ((a - b) ** 2) / ((a + b) ** 2);
  return Math.PI * (a + b) * (1 + (3 * h) / (10 + Math.sqrt(4 - 3 * h)));
}

/**
 * Width in pixels of the body on one image row, median over a small band of rows.
 * With `cx` (the body's centre line, in mask pixels) only the run of foreground pixels that
 * contains the centre counts, so arms held away from the body are left out. Gaps of up to
 * MAX_GAP_FRAC of the width are bridged. `lo` / `hi` cap the run where an arm touches the body.
 * @param {{data: Float32Array|Uint8Array|number[], width: number, height: number}} mask  values 0..1
 */
export function rowWidthPx(mask, yRow, band = 2, cx = null, lo = -Infinity, hi = Infinity) {
  const run = rowRunPx(mask, yRow, band, cx, lo, hi);
  return run ? run.width : null;
}

/** Like rowWidthPx, but also returns where the measured run starts and ends: {width, l, r}. */
export function rowRunPx(mask, yRow, band = 2, cx = null, lo = -Infinity, hi = Infinity) {
  const { data, width: w, height: h } = mask;
  const y0 = Math.max(0, Math.round(yRow) - band), y1 = Math.min(h, Math.round(yRow) + band + 1);
  const maxGap = Math.max(1, Math.round(w * MAX_GAP_FRAC));
  const widths = [], ls = [], rs = [];
  for (let y = y0; y < y1; y++) {
    const off = y * w;
    const on = (x) => data[off + x] > 0.5;
    let first = -1, last = -1, count = 0;
    for (let x = 0; x < w; x++) if (on(x)) { if (first < 0) first = x; last = x; count++; }
    if (count < MIN_MASK_ROW_PIXELS) continue;
    if (cx == null) { widths.push(last - first); ls.push(first); rs.push(last); continue; }
    let c = Math.round(Math.min(w - 1, Math.max(0, cx)));
    if (!on(c)) {
      let d = 1; while (d <= maxGap && !(c - d >= 0 && on(c - d)) && !(c + d < w && on(c + d))) d++;
      if (d > maxGap) continue;
      c = c - d >= 0 && on(c - d) ? c - d : c + d;
    }
    let l = c, r = c;
    for (let x = c - 1, gap = 0; x >= 0; x--) { if (on(x)) { l = x; gap = 0; } else if (++gap > maxGap) break; }
    for (let x = c + 1, gap = 0; x < w; x++) { if (on(x)) { r = x; gap = 0; } else if (++gap > maxGap) break; }
    l = Math.max(l, lo); r = Math.min(r, hi);
    if (r - l + 1 >= MIN_MASK_ROW_PIXELS) { widths.push(r - l); ls.push(l); rs.push(r); }
  }
  return widths.length ? { width: median(widths), l: median(ls), r: median(rs) } : null;
}

export function median(values) {
  const v = values.filter((x) => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/** Linear-interpolated percentile (same as numpy.percentile). */
export function percentile(values, p) {
  const v = values.filter((x) => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const i = (v.length - 1) * p / 100, lo = Math.floor(i), hi = Math.ceil(i);
  return v[lo] + (v[hi] - v[lo]) * (i - lo);
}

// ── pose checks (live guidance) ─────────────────────────────────────────────

/**
 * Is this frame good enough to measure?
 * @param {Array<{x:number,y:number,visibility?:number}>|null} landmarks  normalised 0..1
 * @param {'front'|'side'} view
 * @returns {{ok: boolean, msg: string}}  msg is a key for the UI text
 */
export function checkPose(landmarks, view, w, h) {
  if (!landmarks) return { ok: false, msg: 'whole' };
  const p = (i) => ({ x: landmarks[i].x * w, y: landmarks[i].y * h, v: vis(landmarks[i]) });
  const ls = p(LM.LEFT_SHOULDER), rs = p(LM.RIGHT_SHOULDER), lh = p(LM.LEFT_HIP), rh = p(LM.RIGHT_HIP);
  const la = p(LM.LEFT_ANKLE), ra = p(LM.RIGHT_ANKLE), nose = p(LM.NOSE);
  const torsoOk = [ls, rs, lh, rh].every((q) => q.v > VIS_MIN);
  if (!(la.v > 0.3 || ra.v > 0.3)) return { ok: false, msg: 'feet' };
  const span = Math.abs(la.y - nose.y);
  if (span < h * 0.45) return { ok: false, msg: 'back' };
  if (span > h * 0.98) return { ok: false, msg: 'closer' };
  const shoulderW = Math.abs(ls.x - rs.x);
  if (view === 'front') {
    if (!torsoOk) return { ok: false, msg: 'arms' };
    if (shoulderW < w * 0.10) return { ok: false, msg: 'face' };
    // arms must not touch the torso: hands at the waist were the biggest error source
    const cmPx = 160 / Math.max(1, span);
    const hl = Math.min(lh.x, rh.x), hr = Math.max(lh.x, rh.x), half = (hr - hl) * 0.75, cxh = (hl + hr) / 2;
    for (const wr of [p(LM.LEFT_WRIST), p(LM.RIGHT_WRIST)]) {
      if (wr.v < 0.3 || Math.abs(wr.x - cxh) - half < ARM_GAP_CM / cmPx) return { ok: false, msg: 'arms' };
    }
    return { ok: true, msg: 'hold' };
  }
  if (shoulderW > w * 0.12) return { ok: false, msg: 'turn' };
  if (!(ls.v > VIS_MIN || rs.v > VIS_MIN)) return { ok: false, msg: 'turn' };
  return { ok: true, msg: 'hold' };
}

function vis(l) {
  // Some MediaPipe builds leave visibility empty; treat that as "present".
  return l.visibility == null || Number.isNaN(l.visibility) ? 1 : l.visibility;
}

// ── measuring one frame ─────────────────────────────────────────────────────

function geom(landmarks, w, h, heightCm) {
  const y = (i) => landmarks[i].y * h, x = (i) => landmarks[i].x * w;
  const eyeY = (y(LM.LEFT_EYE) + y(LM.RIGHT_EYE)) / 2;
  const ankleY = (y(LM.LEFT_ANKLE) + y(LM.RIGHT_ANKLE)) / 2;
  const scale = heightCm * (EYE_HEIGHT_FRAC - ANKLE_HEIGHT_FRAC) / Math.max(1, ankleY - eyeY); // cm per px
  const sy = (y(LM.LEFT_SHOULDER) + y(LM.RIGHT_SHOULDER)) / 2, hy = (y(LM.LEFT_HIP) + y(LM.RIGHT_HIP)) / 2;
  const sx = (x(LM.LEFT_SHOULDER) + x(LM.RIGHT_SHOULDER)) / 2, hx = (x(LM.LEFT_HIP) + x(LM.RIGHT_HIP)) / 2;
  const chestY = sy + CHEST_FRAC * (hy - sy), chestX = sx + CHEST_FRAC * (hx - sx);
  // centre line: chest point -> hip point, extended below the hips
  const cxAt = (yy) => chestX + (hx - chestX) * Math.max(0, (yy - chestY) / Math.max(1, hy - chestY));
  // inner edge of each arm at a row: shoulder -> elbow -> wrist (+ hand) moved in by half an arm
  const armLimits = (yy) => {
    const c = cxAt(yy);
    let lo = -Infinity, hi = Infinity;
    for (const ids of [[LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST], [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST]]) {
      const pts = ids.map((i) => ({ x: landmarks[i].x * w, y: landmarks[i].y * h, v: vis(landmarks[i]) }));
      if (pts.some((p) => p.v < 0.3)) continue;
      const [, e, wr] = pts;
      pts[2] = { x: wr.x + HAND_FRAC * (wr.x - e.x), y: wr.y + HAND_FRAC * (wr.y - e.y), v: wr.v };
      for (let k = 0; k < 2; k++) {
        const a = pts[k], b = pts[k + 1];
        if (yy >= Math.min(a.y, b.y) && yy <= Math.max(a.y, b.y) && b.y !== a.y) {
          const ax = a.x + ((yy - a.y) / (b.y - a.y)) * (b.x - a.x);
          const half = (k === 0 ? UPPER_ARM_HALF_CM : FOREARM_HALF_CM) / scale;
          if (ax < c) lo = Math.max(lo, ax + half); else hi = Math.min(hi, ax - half);
          break;
        }
      }
    }
    return { lo, hi };
  };
  return {
    scale, sy, hy, ankleY, cxAt, armLimits,
    lineY: { chest: chestY, waist: sy + WAIST_FRAC * (hy - sy), hip: hy },
    shoulderWpx: Math.abs(x(LM.LEFT_SHOULDER) - x(LM.RIGHT_SHOULDER)),
    hipWpxLm: Math.abs(x(LM.LEFT_HIP) - x(LM.RIGHT_HIP)),
  };
}

/** Measure the three bands on a mask; `arms` = cut arms touching the body (front view). */
function measureBands(G, mask, w, h, arms) {
  const sx = mask.width / w, sy = mask.height / h;   // mask may be smaller than the frame
  const out = {}, segs = [];
  for (const [k, [a, b, n]] of Object.entries(BANDS)) {
    const vals = [];
    for (let i = 0; i < n; i++) {
      const yy = G.sy + (G.hy - G.sy) * (a + (b - a) * i / (n - 1));
      const c = G.cxAt(yy), lim = arms ? G.armLimits(yy) : { lo: -Infinity, hi: Infinity };
      const run = rowRunPx(mask, yy * sy, 1, c * sx, lim.lo * sx, lim.hi * sx);
      if (run) vals.push({ yy, run });
    }
    if (!vals.length) return null;
    const ws = vals.map((v) => v.run.width / sx);
    // chest = median; natural waist = narrowest row (arms / folds only add width); hips = 30th percentile
    out[k] = (k === 'chest' ? median(ws) : k === 'waist' ? Math.min(...ws) : percentile(ws, 30)) * G.scale;
    const pick = k === 'waist' ? vals[ws.indexOf(Math.min(...ws))] : vals[vals.length >> 1];
    segs.push({ y: pick.yy / h, x0: pick.run.l / mask.width, x1: pick.run.r / mask.width });
  }
  out.segs = segs;
  return out;
}

/** Front view -> widths in cm (+ person height in px for auto height). Falls back to landmarks without a mask. */
export function frontWidths(landmarks, mask, w, h, heightCm) {
  const G = geom(landmarks, w, h, heightCm);
  const res = { usedMask: false, shoulder: G.shoulderWpx * G.scale, segs: null, pxH: null, feetY: null };
  const B = mask ? measureBands(G, mask, w, h, true) : null;
  if (B) {
    Object.assign(res, { usedMask: true, chest: B.chest, waist: B.waist, hip: B.hip, segs: B.segs });
    // head top = first mask row in the nose column; floor = ankle + ankle height
    const msx = mask.width / w, msy = mask.height / h;
    const nx = Math.round(Math.min(w - 1, Math.max(0, landmarks[LM.NOSE].x * w)) * msx);
    for (let yy = 0; yy < mask.height; yy++) {
      let hit = false;
      for (let dx = -3; dx <= 3 && !hit; dx++) { const xx = nx + dx; if (xx >= 0 && xx < mask.width && mask.data[yy * mask.width + xx] > 0.5) hit = true; }
      if (hit) { res.pxH = (G.ankleY - yy / msy) / (1 - ANKLE_HEIGHT_FRAC); res.feetY = G.ankleY / h; break; }
    }
  } else {
    const s = G.shoulderWpx * G.scale;
    Object.assign(res, { chest: s * 0.86, waist: s * 0.74, hip: G.hipWpxLm * G.scale * 1.35 });
  }
  return res;
}

/** Side view -> depths in cm, or null without a usable mask. */
export function sideDepths(landmarks, mask, w, h, heightCm) {
  if (!mask) return null;
  const B = measureBands(geom(landmarks, w, h, heightCm), mask, w, h, false);
  return B ? { chest: B.chest, waist: B.waist, hip: B.hip, segs: B.segs } : null;
}

/**
 * Auto height for a FIXED camera: the person stands on a floor marker, and one calibration
 * scan of someone with a known height gives cm per pixel. Feet lower in the frame = closer.
 * @param {Array} fronts  frontWidths() results (any height used)
 * @param {{cmPerPx:number, feetY?:number, feetK?:number}} cal
 */
export function predictHeight(fronts, cal) {
  if (!cal || !cal.cmPerPx) return null;
  const px = median(fronts.map((f) => f.pxH)), fy = median(fronts.map((f) => f.feetY));
  if (!px) return null;
  const corr = 1 - (cal.feetK || 0) * (fy - (cal.feetY ?? fy));
  const cm = Math.round(px * cal.cmPerPx * corr);
  return cm >= 140 && cm <= 210 ? cm : null;
}

/** Calibration for predictHeight from the fronts of one scan of a person of known height. */
export function heightCalibration(fronts, knownCm) {
  const px = median(fronts.map((f) => f.pxH)), fy = median(fronts.map((f) => f.feetY));
  return px ? { cmPerPx: knownCm / px, feetY: fy, feetK: 0 } : null;
}

/**
 * Combine the per-frame results (median of each value) into circumferences.
 * @param {Array} fronts  frontWidths() results (measured with heightCm)
 * @param {Array} sides   sideDepths() results (may be empty)
 * @param {object} calib  multiplicative corrections {chest, waist, hip}
 */
export function combine(fronts, sides, heightCm, calib = DEFAULT_CALIBRATION) {
  if (!fronts.length) return null;
  const med = (arr, k) => median(arr.map((f) => f[k]));
  const W = { chest: med(fronts, 'chest'), waist: med(fronts, 'waist'), hip: med(fronts, 'hip'), shoulder: med(fronts, 'shoulder') };
  const usedMask = fronts.filter((f) => f.usedMask).length * 2 >= fronts.length;
  const goodSides = sides.filter(Boolean);
  const usedSide = goodSides.length > 0;
  const D = usedSide ? { chest: med(goodSides, 'chest'), waist: med(goodSides, 'waist'), hip: med(goodSides, 'hip') } : null;
  const flags = [], circ = {}, depth = {};
  for (const k of ['chest', 'waist', 'hip']) {
    const [lo, hi] = DEPTH_RATIO[k];
    let r = DEFAULT_DEPTH_RATIO[k];
    if (usedSide) {
      r = D[k] / Math.max(1e-3, W[k]);
      if (r < lo || r > hi) flags.push(`${k}_depth_clamped(${r.toFixed(2)})`);
      r = Math.min(hi, Math.max(lo, r));
    }
    depth[k] = W[k] * r;
    let c = ellipseCircumference(W[k], depth[k]) * SHAPE_FACTOR[k] * ((calib && calib[k]) || 1);
    const [blo, bhi] = STATURE_BOUNDS[k];
    if (c < blo * heightCm || c > bhi * heightCm) { flags.push(`${k}_out_of_range(${Math.round(c)})`); c = Math.min(bhi * heightCm, Math.max(blo * heightCm, c)); }
    circ[k] = c;
  }
  let quality = usedMask && usedSide && fronts.length >= 4 ? 'high' : usedMask || usedSide ? 'medium' : 'low';
  if (flags.some((f) => f.includes('out_of_range'))) quality = 'low';
  else if (flags.length && quality === 'high') quality = 'medium';
  const r1 = (v) => Math.round(v * 10) / 10;
  return {
    height_cm: heightCm,
    shoulder_cm: r1(W.shoulder),
    chest_cm: r1(circ.chest), waist_cm: r1(circ.waist), hip_cm: r1(circ.hip),
    quality,
    frames: { front: fronts.length, side: goodSides.length },
    raw: { width: Object.fromEntries(Object.entries(W).map(([k, v]) => [k, r1(v)])), depth: Object.fromEntries(Object.entries(depth).map(([k, v]) => [k, r1(v)])), flags },
  };
}

// ── size chart lookup ───────────────────────────────────────────────────────

function rangeScore(value, lo, hi) {
  const mid = (lo + hi) / 2, half = Math.max(0.5, (hi - lo) / 2);
  return Math.abs(value - mid) / half;
}

function pickFromRanges(value, ranges) {
  let best = null, bestScore = Infinity;
  for (const [size, [lo, hi]] of Object.entries(ranges)) {
    const s = rangeScore(value, lo, hi);
    if (s < bestScore) { best = size; bestScore = s; }
  }
  const order = SIZE_ORDER.filter((s) => s in ranges);
  const idx = order.indexOf(best);
  const [lo, hi] = ranges[best];
  const edge = Math.min(value - lo, hi - value);
  let neighbor = null;
  if (edge < 2.0) {
    if (value < (lo + hi) / 2 && idx > 0) neighbor = order[idx - 1];
    else if (idx < order.length - 1) neighbor = order[idx + 1];
  }
  return { size: best, neighbor, edge };
}

/**
 * @param {{chest_cm?:number, waist_cm?:number, hip_cm?:number, quality?:string}} m
 * @param {'tops'|'bottoms'|'one-pieces'} category
 * @param {'men'|'women'|object} gender  a key of SIZE_CHARTS, or a chart object of your own
 * @returns {{size:string|null, alternative:string|null, confidence:string, basis:string|null}}
 */
export function recommendSize(m, category, gender = 'women') {
  const charts = typeof gender === 'string' ? (SIZE_CHARTS[gender] || SIZE_CHARTS.women) : gender;
  const chart = charts[category] || charts.tops;
  const quality = m.quality || 'medium';
  let size, neighbor, edge, basis;
  if (category === 'bottoms') {
    if (m.waist_cm == null) return none();
    ({ size, neighbor, edge } = pickFromRanges(m.waist_cm, mapRanges(chart, 'waist')));
    basis = `waist ${Math.round(m.waist_cm)} cm`;
  } else if (category === 'one-pieces') {
    if (m.chest_cm == null || m.hip_cm == null) return none();
    const c = pickFromRanges(m.chest_cm, mapRanges(chart, 'chest'));
    const hp = pickFromRanges(m.hip_cm, mapRanges(chart, 'hip'));
    size = SIZE_ORDER.indexOf(c.size) >= SIZE_ORDER.indexOf(hp.size) ? c.size : hp.size;
    neighbor = size === c.size ? hp.size : c.size;
    if (neighbor === size) neighbor = null;
    edge = Math.min(c.edge, hp.edge);
    basis = `chest ${Math.round(m.chest_cm)} cm / hip ${Math.round(m.hip_cm)} cm`;
  } else {
    if (m.chest_cm == null) return none();
    ({ size, neighbor, edge } = pickFromRanges(m.chest_cm, chart));
    basis = `chest ${Math.round(m.chest_cm)} cm`;
  }
  const confidence = quality === 'high' && edge > 2.5 ? 'high' : quality === 'low' || edge < 1.0 ? 'low' : 'medium';
  return { size, alternative: neighbor, confidence, basis };
}

/** French trouser number from the waist (36..48 men, 32..48 women; above 48 -> "48+"). */
export function trouserFR(waistCm, gender = 'men') {
  if (waistCm == null) return null;
  const ranges = TROUSER_FR[gender] || TROUSER_FR.men;
  let best = null, bestScore = Infinity;
  for (const [n, [lo, hi]] of Object.entries(ranges)) { const s = rangeScore(waistCm, lo, hi); if (s < bestScore) { best = Number(n); bestScore = s; } }
  return best > RETAIL_MAX ? `${RETAIL_MAX}+` : best;
}

/** Everything for one person: tops / bottoms letter / trouser number for men and women. */
export function recommendSizeSet(m) {
  const out = {};
  for (const g of ['men', 'women']) {
    out[g] = { tops: recommendSize(m, 'tops', g).size, bottoms: recommendSize(m, 'bottoms', g).size, trousers_fr: trouserFR(m.waist_cm, g) };
  }
  return out;
}

function mapRanges(chart, key) {
  return Object.fromEntries(Object.entries(chart).map(([s, v]) => [s, v[key]]));
}

function none() {
  return { size: null, alternative: null, confidence: 'none', basis: null };
}

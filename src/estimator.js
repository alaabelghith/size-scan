/**
 * estimator.js — Size Scan measurement maths.
 *
 * Pure functions, no browser APIs, so they run (and are tested) in Node too.
 * Same method as python/size_estimator.py, the version that runs in the
 * Fashion Mirror kiosk:
 *
 *   1. Scale. The user's height and the eye-to-ankle distance in the image give
 *      centimetres per pixel (eyes sit about 7 % of standing height below the top
 *      of the head).
 *   2. Widths. On the front view, the silhouette from the segmentation mask is
 *      measured at a chest line, a waist line and the hip line, placed between the
 *      shoulder and hip landmarks. Only the body counts: arms held away from it
 *      are separated by background and left out.
 *   3. Depths. The same three lines are measured on the side view.
 *   4. Circumference. Each cross-section is treated as an ellipse (width x depth),
 *      using Ramanujan's approximation for its perimeter.
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
export const EYE_TO_TOP_FRACTION = 0.07; // eyes sit ~7 % of standing height below the top of the head
export const CHEST_FRAC = 0.18;         // chest line, as a fraction of the shoulder -> hip span
export const WAIST_FRAC = 0.62;         // waist line
export const MIN_MASK_ROW_PIXELS = 8;   // fewer foreground pixels than this = unreliable row
export const DEPTH_RATIO = 0.62;        // body depth / width when no side view is available
export const UPPER_ARM_HALF_CM = 4.5;  // half the thickness of an upper arm / forearm, used where an arm
export const FOREARM_HALF_CM = 3.5;    // touches the body in the front view and has to be cut off
export const MAX_GAP_FRAC = 0.004;      // background gap (fraction of image width) still counted as body

/**
 * Demo size chart (cm). Replace with your own brand's chart: sizes run small to
 * large, and every category needs the same size names.
 */
export const SIZE_CHARTS = {
  tops: { // chest circumference
    S: [82, 90], M: [90, 98], L: [98, 106], XL: [106, 116],
  },
  bottoms: { // waist decides, hip shown
    S: { waist: [66, 74], hip: [88, 96] },
    M: { waist: [74, 82], hip: [96, 104] },
    L: { waist: [82, 90], hip: [104, 112] },
    XL: { waist: [90, 100], hip: [112, 122] },
  },
  'one-pieces': { // the larger of the chest and hip sizes
    S: { chest: [82, 90], hip: [88, 96] },
    M: { chest: [90, 98], hip: [96, 104] },
    L: { chest: [98, 106], hip: [104, 112] },
    XL: { chest: [106, 116], hip: [112, 122] },
  },
};
export const SIZE_ORDER = ['S', 'M', 'L', 'XL'];

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
 *
 * With `cx` (the body's centre line, in mask pixels) only the run of foreground
 * pixels that contains the centre counts, so arms held slightly away from the
 * body are left out. Gaps of up to `MAX_GAP_FRAC` of the image width are bridged
 * (mask noise). Without `cx` it falls back to first-to-last foreground pixel.
 * `lo` / `hi` cap the run where an arm touches the body (see armLimits).
 * Returns null if the row is empty.
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
    if (!on(c)) { // centre fell on background: take the nearest foreground pixel within a gap
      let d = 1; while (d <= maxGap && !(c - d >= 0 && on(c - d)) && !(c + d < w && on(c + d))) d++;
      if (d > maxGap) continue;
      c = c - d >= 0 && on(c - d) ? c - d : c + d;
    }
    let l = c, r = c;
    for (let x = c - 1, gap = 0; x >= 0; x--) { if (on(x)) { l = x; gap = 0; } else if (++gap > maxGap) break; }
    for (let x = c + 1, gap = 0; x < w; x++) { if (on(x)) { r = x; gap = 0; } else if (++gap > maxGap) break; }
    l = Math.max(l, lo); r = Math.min(r, hi);   // arms merged with the body: cut at the arm's inner edge
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

// ── pose checks (live guidance) ─────────────────────────────────────────────

/**
 * Is this frame good enough to measure?
 * @param {Array<{x:number,y:number,visibility?:number}>|null} landmarks  normalised 0..1
 * @param {'front'|'side'} view
 * @param {number} w @param {number} h  image size in pixels
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
    return { ok: true, msg: 'hold' };
  }
  if (shoulderW > w * 0.09) return { ok: false, msg: 'turn' };
  if (!(ls.v > VIS_MIN || rs.v > VIS_MIN)) return { ok: false, msg: 'turn' };
  return { ok: true, msg: 'hold' };
}

function vis(l) {
  // Some MediaPipe builds leave visibility empty; treat that as "present".
  return l.visibility == null || Number.isNaN(l.visibility) ? 1 : l.visibility;
}

// ── measuring one frame ─────────────────────────────────────────────────────

function lines(landmarks, w, h, heightCm) {
  const y = (i) => landmarks[i].y * h, x = (i) => landmarks[i].x * w;
  const eyeY = (y(LM.LEFT_EYE) + y(LM.RIGHT_EYE)) / 2;
  const ankleY = (y(LM.LEFT_ANKLE) + y(LM.RIGHT_ANKLE)) / 2;
  const totalPx = Math.max(1, ankleY - eyeY) / (1 - EYE_TO_TOP_FRACTION);
  const shoulderY = (y(LM.LEFT_SHOULDER) + y(LM.RIGHT_SHOULDER)) / 2;
  const hipY = (y(LM.LEFT_HIP) + y(LM.RIGHT_HIP)) / 2;
  const shoulderX = (x(LM.LEFT_SHOULDER) + x(LM.RIGHT_SHOULDER)) / 2;
  const hipX = (x(LM.LEFT_HIP) + x(LM.RIGHT_HIP)) / 2;
  const cxAt = (f) => shoulderX + f * (hipX - shoulderX); // body centre line, f = 0 shoulders .. 1 hips
  const scale = heightCm / totalPx;
  // Inner edge of each arm at a given row: the arm's centre line (shoulder -> elbow -> wrist)
  // moved inwards by half an arm's thickness. Null where the arm is not at that height.
  const armEdge = (ids, yy) => {
    const pts = ids.map((i) => landmarks[i]);
    if (pts.some((p) => vis(p) < 0.3)) return null;
    for (let k = 0; k < 2; k++) {
      const a = pts[k], b = pts[k + 1], ay = a.y * h, by = b.y * h;
      if (yy >= Math.min(ay, by) && yy <= Math.max(ay, by) && by !== ay) {
        const ax = a.x * w + ((yy - ay) / (by - ay)) * (b.x - a.x) * w;
        const half = (k === 0 ? UPPER_ARM_HALF_CM : FOREARM_HALF_CM) / scale;
        const centre = cxAt(Math.min(1, Math.max(0, (yy - shoulderY) / Math.max(1, hipY - shoulderY))));
        return ax < centre ? ax + half : ax - half;
      }
    }
    return null;
  };
  const limits = (yy) => {
    const e = [armEdge([LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST], yy), armEdge([LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST], yy)];
    const c = cxAt(Math.min(1, Math.max(0, (yy - shoulderY) / Math.max(1, hipY - shoulderY))));
    let lo = -Infinity, hi = Infinity;
    for (const v of e) { if (v == null) continue; if (v < c) lo = Math.max(lo, v); else hi = Math.min(hi, v); }
    return { lo, hi };
  };
  const chestY = shoulderY + CHEST_FRAC * (hipY - shoulderY), waistY = shoulderY + WAIST_FRAC * (hipY - shoulderY);
  return {
    scale, // cm per pixel
    shoulderY, hipY,
    limits: { chest: limits(chestY), waist: limits(waistY), hip: limits(hipY) },
    cx: { chest: cxAt(CHEST_FRAC), waist: cxAt(WAIST_FRAC), hip: cxAt(1) },
    chestY: shoulderY + CHEST_FRAC * (hipY - shoulderY),
    waistY: shoulderY + WAIST_FRAC * (hipY - shoulderY),
    shoulderWpx: Math.abs(x(LM.LEFT_SHOULDER) - x(LM.RIGHT_SHOULDER)),
    hipWpxLm: Math.abs(x(LM.LEFT_HIP) - x(LM.RIGHT_HIP)),
  };
}

/** Front view -> widths in cm. Falls back to landmarks if the mask is missing. */
export function frontWidths(landmarks, mask, w, h, heightCm) {
  const L = lines(landmarks, w, h, heightCm);
  let chest = null, waist = null, hip = null, usedMask = false, segs = null;
  if (mask) {
    const sx = mask.width / w, sy = mask.height / h; // mask may be smaller than the frame
    const at = (y, cx, lim) => rowRunPx(mask, y * sy, 2, cx * sx, lim.lo * sx, lim.hi * sx);
    const c = at(L.chestY, L.cx.chest, L.limits.chest), wa = at(L.waistY, L.cx.waist, L.limits.waist), hi = at(L.hipY, L.cx.hip, L.limits.hip);
    if (c && wa && hi) {
      chest = c.width / sx; waist = wa.width / sx; hip = hi.width / sx; usedMask = true;
      segs = [[L.chestY, c], [L.waistY, wa], [L.hipY, hi]].map(([y, q]) => ({ y: y / h, x0: q.l / mask.width, x1: q.r / mask.width }));
    }
  }
  if (!usedMask) {
    chest = L.shoulderWpx - CHEST_FRAC * (L.shoulderWpx - L.hipWpxLm);
    waist = L.shoulderWpx - WAIST_FRAC * (L.shoulderWpx - L.hipWpxLm);
    hip = L.hipWpxLm;
  }
  return {
    usedMask,
    shoulder: L.shoulderWpx * L.scale,
    chest: chest * L.scale, waist: waist * L.scale, hip: hip * L.scale,
    segs, // where each width was measured, normalised 0..1 (for drawing)
  };
}

/** Side view -> depths in cm, or null without a usable mask. */
export function sideDepths(landmarks, mask, w, h, heightCm) {
  if (!mask) return null;
  const L = lines(landmarks, w, h, heightCm);
  const sx = mask.width / w, sy = mask.height / h;
  const c = rowRunPx(mask, L.chestY * sy, 2, L.cx.chest * sx);
  const wa = rowRunPx(mask, L.waistY * sy, 2, L.cx.waist * sx);
  const hi = rowRunPx(mask, L.hipY * sy, 2, L.cx.hip * sx);
  if (!(c && wa && hi)) return null;
  return {
    chest: (c.width / sx) * L.scale, waist: (wa.width / sx) * L.scale, hip: (hi.width / sx) * L.scale,
    segs: [[L.chestY, c], [L.waistY, wa], [L.hipY, hi]].map(([y, q]) => ({ y: y / h, x0: q.l / mask.width, x1: q.r / mask.width })),
  };
}

/**
 * Combine the per-frame results (median of each value) into circumferences.
 * @param {Array} fronts  frontWidths() results
 * @param {Array} sides   sideDepths() results (may be empty)
 */
export function combine(fronts, sides, heightCm) {
  if (!fronts.length) return null;
  const med = (arr, k) => median(arr.map((f) => f[k]));
  const fw = { chest: med(fronts, 'chest'), waist: med(fronts, 'waist'), hip: med(fronts, 'hip'), shoulder: med(fronts, 'shoulder') };
  const usedMask = fronts.filter((f) => f.usedMask).length * 2 >= fronts.length;
  const goodSides = sides.filter(Boolean);
  const usedSide = goodSides.length > 0;
  const sd = usedSide
    ? { chest: med(goodSides, 'chest'), waist: med(goodSides, 'waist'), hip: med(goodSides, 'hip') }
    : { chest: fw.chest * DEPTH_RATIO, waist: fw.waist * DEPTH_RATIO, hip: fw.hip * DEPTH_RATIO };
  const quality = usedMask && usedSide ? 'high' : usedMask || usedSide ? 'medium' : 'low';
  const r1 = (v) => Math.round(v * 10) / 10;
  return {
    height_cm: heightCm,
    shoulder_cm: r1(fw.shoulder),
    chest_cm: r1(ellipseCircumference(fw.chest, sd.chest)),
    waist_cm: r1(ellipseCircumference(fw.waist, sd.waist)),
    hip_cm: r1(ellipseCircumference(fw.hip, sd.hip)),
    quality,
    frames: { front: fronts.length, side: goodSides.length },
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
 * @returns {{size:string|null, alternative:string|null, confidence:string, basis:string|null}}
 */
export function recommendSize(m, category, charts = SIZE_CHARTS) {
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

function mapRanges(chart, key) {
  return Object.fromEntries(Object.entries(chart).map(([s, v]) => [s, v[key]]));
}

function none() {
  return { size: null, alternative: null, confidence: 'none', basis: null };
}

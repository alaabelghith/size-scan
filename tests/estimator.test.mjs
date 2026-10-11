// Run: node tests/estimator.test.mjs   (same checks as python/size_estimator.py --selftest)
import assert from 'node:assert/strict';
import {
  ellipseCircumference, rowWidthPx, recommendSize, frontWidths, combine, checkPose, LM,
} from '../src/estimator.js';

// 1. a circle of radius 15 cm
assert.ok(Math.abs(ellipseCircumference(30, 30) - 2 * Math.PI * 15) < 0.01);

// 2. silhouette width on a synthetic mask (40 px band)
const W = 100, H = 200, data = new Float32Array(W * H);
for (let y = 50; y < 150; y++) for (let x = 30; x < 70; x++) data[y * W + x] = 1;
const mask = { data, width: W, height: H };
const w = rowWidthPx(mask, 100);
assert.ok(w >= 38 && w <= 40, `row width ${w}`);
assert.equal(rowWidthPx(mask, 10), null);

// 3. size chart lookup
assert.equal(recommendSize({ chest_cm: 94, quality: 'high' }, 'tops').size, 'M');
const b = recommendSize({ waist_cm: 89, quality: 'medium' }, 'bottoms');
assert.equal(b.size, 'L'); assert.equal(b.alternative, 'XL');
assert.equal(recommendSize({ chest_cm: 100, hip_cm: 118, quality: 'medium' }, 'one-pieces').size, 'XL');

// 4. full pipeline on a synthetic ~178 cm person (same numbers as the Python self-test)
const FW = 400, FH = 960;
const pts = Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
const set = (i, x, y) => { pts[i] = { x: x / FW, y: y / FH, visibility: 0.9 }; };
set(LM.NOSE, 200, 120); set(LM.LEFT_EYE, 190, 110); set(LM.RIGHT_EYE, 210, 110);
set(LM.LEFT_SHOULDER, 105, 220); set(LM.RIGHT_SHOULDER, 295, 220);
set(LM.LEFT_HIP, 125, 480); set(LM.RIGHT_HIP, 275, 480);
set(LM.LEFT_ANKLE, 170, 900); set(LM.RIGHT_ANKLE, 230, 900);
const body = new Float32Array(FW * FH);
for (let y = 200; y < 500; y++) {
  const half = Math.trunc(100 - 15 * ((y - 200) / 300));
  for (let x = 200 - half; x < 200 + half; x++) body[y * FW + x] = 1;
}
const fw = frontWidths(pts, { data: body, width: FW, height: FH }, FW, FH, 178);
const m = combine([fw], [], 178);
// Python reference: {'shoulder_cm': 39.8, 'chest_cm': 103.3, 'waist_cm': 96.8, 'hip_cm': 92.4, 'quality': 'medium'}
assert.deepEqual([m.shoulder_cm, m.chest_cm, m.waist_cm, m.hip_cm, m.quality], [39.8, 103.3, 96.8, 92.4, 'medium']);

// 5. pose guidance
assert.equal(checkPose(pts, 'front', FW, FH).ok, true);
assert.equal(checkPose(pts, 'side', FW, FH).msg, 'turn');
assert.equal(checkPose(null, 'front', FW, FH).msg, 'whole');

console.log('all estimator tests passed', m);

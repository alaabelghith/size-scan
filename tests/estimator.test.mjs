// Run: node tests/estimator.test.mjs   (same checks as python/size_estimator.py --selftest)
import assert from 'node:assert/strict';
import {
  ellipseCircumference, rowWidthPx, recommendSize, recommendSizeSet, trouserFR, frontWidths, combine,
  checkPose, predictHeight, heightCalibration, fitHorizon, heightFrom, percentile, LM,
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
assert.equal(percentile([1, 2, 3, 4], 30), 1.9);   // numpy.percentile

// 3. size charts
assert.equal(recommendSize({ chest_cm: 94, quality: 'high' }, 'tops', 'women').size, 'M');
assert.equal(recommendSize({ chest_cm: 112, quality: 'high' }, 'tops', 'men').size, 'XL');   // Primark: XL 109-118
assert.equal(trouserFR(96, 'men'), 48);            // Celio: 48 ~ 96 cm
assert.equal(trouserFR(84, 'men'), 42);
assert.equal(trouserFR(110, 'men'), '48+');         // mass retail stops at 48
assert.equal(trouserFR(70, 'women'), 36);
const set = recommendSizeSet({ chest_cm: 100, waist_cm: 84, hip_cm: 104, quality: 'high' });
assert.ok(set.men && set.women);

// 4. full pipeline on a synthetic ~178 cm person (same numbers as the Python self-test)
const FW = 400, FH = 960;
const pts = Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
const put = (i, x, y) => { pts[i] = { x: x / FW, y: y / FH, visibility: 0.9 }; };
put(LM.NOSE, 200, 120); put(LM.LEFT_EYE, 190, 110); put(LM.RIGHT_EYE, 210, 110);
put(LM.LEFT_SHOULDER, 105, 220); put(LM.RIGHT_SHOULDER, 295, 220);
put(LM.LEFT_HIP, 125, 480); put(LM.RIGHT_HIP, 275, 480);
put(LM.LEFT_ANKLE, 170, 900); put(LM.RIGHT_ANKLE, 230, 900);
const body = new Float32Array(FW * FH);
for (let y = 200; y < 500; y++) {
  const half = Math.trunc(100 - 15 * ((y - 200) / 300));
  for (let x = 200 - half; x < 200 + half; x++) body[y * FW + x] = 1;
}
const fw = frontWidths(pts, { data: body, width: FW, height: FH }, FW, FH, 178);
const m = combine([fw], [], 178, {});
// Python reference (calibration off): shoulder 38.4, chest 110.1, waist 99.7, hip 97.7, medium
assert.deepEqual([m.shoulder_cm, m.chest_cm, m.waist_cm, m.hip_cm, m.quality], [38.4, 110.1, 99.7, 97.7, 'medium']);
assert.ok(m.chest_cm > 60 && m.chest_cm < 140);

// 5. auto height: calibrate on this person, predict it back
const cal = heightCalibration([fw], 178);
assert.equal(predictHeight([fw], cal), 178);
// ground model: one person seen at several distances (horizon at 0.45) -> the distance cancels
const walk = Array.from({ length: 20 }, (_, i) => { const fy = 0.75 + i * 0.005; return { pxHn: 2.0 * (fy - 0.45), pxH: 2000 * (fy - 0.45), feetY: fy }; });
assert.ok(Math.abs(fitHorizon(walk) - 0.45) < 1e-6);
const cal2 = heightCalibration([walk[5]], 183, walk);
assert.equal(Math.round(heightFrom(walk[15], cal2)), 183);   // closer to the camera, same height

// 6. pose guidance: front needs the arms away from the body
assert.equal(checkPose(pts, 'front', FW, FH).msg, 'arms');          // no wrists visible
put(LM.LEFT_WRIST, 20, 500); put(LM.RIGHT_WRIST, 380, 500);
assert.equal(checkPose(pts, 'front', FW, FH).ok, true);
assert.equal(checkPose(pts, 'side', FW, FH).msg, 'turn');
assert.equal(checkPose(null, 'front', FW, FH).msg, 'whole');

console.log('all estimator tests passed', m);

/**
 * app.js — Size Scan page: camera, MediaPipe Pose, guidance, result.
 * The measuring maths lives in estimator.js.
 */
import { checkPose, frontWidths, sideDepths, combine, recommendSize } from './estimator.js';
import { T } from './i18n.js';

const MP_VERSION = '0.10.35';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';
const STABLE_FRAMES = 12;      // consecutive good frames per pose (same as the kiosk)
const PROC_H = 640;            // frames are analysed at this height
const SKIP_SIDE_AFTER_MS = 20000;

const $ = (id) => document.getElementById(id);
const S = {
  lang: 'en', height: 175, landmarker: null, mode: null,
  stream: null, rotate: 0, mirror: true, running: false, raf: 0,
  stage: 'front', stable: 0, fronts: [], sides: [], frontVis: null, sideVis: null, stageStart: 0, pause: 0,
};
const t = (k) => T[S.lang][k] ?? T.en[k];

// ── language, theme, height ────────────────────────────────────────────────
function setLang(l) {
  S.lang = T[l] ? l : 'en';
  document.documentElement.lang = S.lang;
  document.documentElement.dir = S.lang === 'ar' ? 'rtl' : 'ltr';
  document.querySelectorAll('[data-t]').forEach((el) => { el.textContent = t(el.dataset.t); });
  document.querySelectorAll('.langs button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.l === S.lang)));
  try { localStorage.setItem('ss.lang', S.lang); } catch {}
  if (S.lastResult) renderResult(S.lastResult);
}
function setTheme(th) {
  if (th) document.documentElement.dataset.theme = th; else delete document.documentElement.dataset.theme;
  try { th ? localStorage.setItem('ss.theme', th) : localStorage.removeItem('ss.theme'); } catch {}
}
function currentTheme() {
  return document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
}
function setHeight(v) {
  S.height = Math.max(140, Math.min(210, Math.round(Number(v) || 175)));
  $('heightVal').value = S.height;
  try { localStorage.setItem('ss.height', S.height); } catch {}
}

function show(id) {
  for (const s of document.querySelectorAll('.screen')) s.hidden = s.id !== id;
  window.scrollTo({ top: 0 });
}
function status(msg, kind = '') {
  const el = $('status'); el.textContent = msg; el.dataset.kind = kind;
}

// ── MediaPipe ──────────────────────────────────────────────────────────────
async function loadModel(runningMode) {
  if (!S.landmarker) {
    $('loadNote').hidden = false;
    if (!$('photos').hidden) photoMsg(t('loading'));
    const { PoseLandmarker, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
    const files = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode, numPoses: 1, outputSegmentationMasks: true,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    // CPU: reading the segmentation mask back from a GPU texture fails on some
    // devices (float readback), and the full model runs in real time on CPU anyway.
    const delegate = new URLSearchParams(location.search).get('gpu') === '1' ? 'GPU' : 'CPU';
    S.landmarker = await PoseLandmarker.createFromOptions(files, opts(delegate));
    S.mode = runningMode;
    $('loadNote').hidden = true;
  }
  if (S.mode !== runningMode) { await S.landmarker.setOptions({ runningMode }); S.mode = runningMode; }
  return S.landmarker;
}

/** Run the model; returns {landmarks, mask} copied out of MediaPipe's buffers. */
function detect(source, ts) {
  let out = { landmarks: null, mask: null };
  const cb = (r) => {
    const lm = r.landmarks && r.landmarks[0];
    let mask = null;
    if (lm && r.segmentationMasks && r.segmentationMasks[0]) {
      const m = r.segmentationMasks[0];
      mask = { data: m.getAsFloat32Array().slice(), width: m.width, height: m.height };
    }
    out = { landmarks: lm ? lm.map((p) => ({ x: p.x, y: p.y, visibility: p.visibility })) : null, mask };
  };
  if (S.mode === 'VIDEO') S.landmarker.detectForVideo(source, ts, cb);
  else S.landmarker.detect(source, cb);
  return out;
}

// ── camera ─────────────────────────────────────────────────────────────────
async function startCamera(deviceId) {
  stopCamera();
  const video = $('video');
  const constraints = { audio: false, video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
    : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } };
  S.stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = S.stream;
  await video.play();
  const track = S.stream.getVideoTracks()[0];
  const facing = track.getSettings().facingMode;
  S.mirror = facing !== 'environment';
  // camera list (labels are only available after permission)
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const sel = $('camSel');
  sel.innerHTML = '';
  cams.forEach((c, i) => { const o = document.createElement('option'); o.value = c.deviceId; o.textContent = c.label || `${t('camera')} ${i + 1}`; sel.append(o); });
  sel.value = track.getSettings().deviceId || '';
  $('camSelWrap').hidden = cams.length < 2;
  // a phone used as a webcam (DroidCam and similar) usually arrives sideways
  if (!deviceId) {
    const droid = cams.find((c) => /droidcam|iriun|epoccam|camo/i.test(c.label));
    if (droid && droid.deviceId !== sel.value) return startCamera(droid.deviceId);
  }
  if (/droidcam|iriun|epoccam|camo/i.test(track.label) && !S.rotateTouched) S.rotate = 90;
}
function stopCamera() {
  if (S.stream) S.stream.getTracks().forEach((tr) => tr.stop());
  S.stream = null;
}

const mult4 = (v) => Math.max(4, Math.round(v / 4) * 4);

/** Draw the (rotated) video frame into the processing canvas. */
function grabFrame() {
  const v = $('video'), c = $('proc');
  const vw = v.videoWidth, vh = v.videoHeight;
  if (!vw || !vh) return null;
  const rot = S.rotate % 180 !== 0;
  const fw = rot ? vh : vw, fh = rot ? vw : vh;
  const k = PROC_H / fh;
  c.width = mult4(fw * k); c.height = PROC_H;   // MediaPipe's CPU path aborts on widths that are not a multiple of 4
  const g = c.getContext('2d', { willReadFrequently: false });
  g.save();
  g.translate(c.width / 2, c.height / 2);
  g.rotate((S.rotate * Math.PI) / 180);
  g.drawImage(v, -vw * k / 2, -vh * k / 2, vw * k, vh * k);
  g.restore();
  return c;
}

// ── live scan loop ─────────────────────────────────────────────────────────
async function startScan() {
  show('scan');
  resetScan();
  status('');
  try { await loadModel('VIDEO'); } catch (e) { console.error(e); status(t('modelError'), 'err'); return; }
  try { await startCamera(); } catch (e) { console.error(e); status(t('camError'), 'err'); return; }
  S.running = true;
  loop();
}
function resetScan() {
  Object.assign(S, { stage: 'front', stable: 0, fronts: [], sides: [], frontVis: null, sideVis: null, stageStart: performance.now(), pause: 0 });
  setStage('front');
  $('skipSide').hidden = true;
  setProgress(0);
}
function stopScan() {
  S.running = false;
  cancelAnimationFrame(S.raf);
  stopCamera();
}
function setStage(stage) {
  S.stage = stage;
  document.querySelectorAll('.steps li').forEach((li) => {
    li.dataset.state = li.dataset.s === stage ? 'now' : (stage === 'side' && li.dataset.s === 'front') ? 'done' : '';
  });
}
function setProgress(p) { $('bar').style.setProperty('--p', String(p)); }

let lastTs = 0;
function loop() {
  if (!S.running) return;
  S.raf = requestAnimationFrame(loop);
  const now = performance.now();
  if (now - lastTs < 50) return;         // ~20 analyses per second is plenty
  lastTs = now;
  const frame = grabFrame();
  if (!frame) return;
  const { landmarks, mask } = detect(frame, now);
  draw(frame, landmarks, mask);
  if (now < S.pause) return;

  const w = frame.width, h = frame.height;
  const chk = checkPose(landmarks, S.stage, w, h);
  if (!chk.ok) {
    S.stable = 0;
    if (S.stage === 'front') S.fronts = []; else S.sides = [];
    status(t(chk.msg));
    setProgress(0);
  } else {
    S.stable++;
    status(t('hold'), 'ok');
    if (S.stage === 'front') S.fronts.push(frontWidths(landmarks, mask, w, h, S.height));
    else S.sides.push(sideDepths(landmarks, mask, w, h, S.height));
    setProgress(S.stable / STABLE_FRAMES);
  }
  if (S.stage === 'side' && now - S.stageStart > SKIP_SIDE_AFTER_MS) $('skipSide').hidden = false;

  if (S.stable >= STABLE_FRAMES) {
    if (S.stage === 'front') {
      S.frontVis = snapshotSilhouette(mask, w, h, S.fronts.at(-1).segs);
      S.stable = 0; S.stageStart = now; S.pause = now + 1400;
      setStage('side'); status(t('frontOk'), 'ok'); setProgress(0);
      pulse();
    } else {
      const sd = S.sides.at(-1);
      S.sideVis = sd ? snapshotSilhouette(mask, w, h, sd.segs) : null;
      finish();
    }
  }
}

function finish(skipSide = false) {
  stopScan();
  status(t('measuring'));
  const m = combine(S.fronts, skipSide ? [] : S.sides, S.height);
  if (!m) { status(t('failed'), 'err'); return; }
  S.lastResult = { m, frontVis: S.frontVis, sideVis: skipSide ? null : S.sideVis, note: '' };
  renderResult(S.lastResult);
  show('result');
}

function pulse() {
  const el = $('stage');
  el.animate([{ opacity: 1 }, { opacity: 0.35 }, { opacity: 1 }], { duration: 420, easing: 'cubic-bezier(0.23,1,0.32,1)' });
}

// ── drawing ────────────────────────────────────────────────────────────────
const BONES = [[11, 12], [11, 23], [12, 24], [23, 24], [11, 13], [13, 15], [12, 14], [14, 16], [23, 25], [25, 27], [24, 26], [26, 28]];
const tint = document.createElement('canvas');

function draw(frame, landmarks, mask) {
  const c = $('view');
  if (c.width !== frame.width || c.height !== frame.height) { c.width = frame.width; c.height = frame.height; }
  const g = c.getContext('2d');
  g.drawImage(frame, 0, 0);
  if (mask) {
    tint.width = mask.width; tint.height = mask.height;
    const tg = tint.getContext('2d');
    const img = tg.createImageData(mask.width, mask.height);
    for (let i = 0; i < mask.data.length; i++) {
      const a = mask.data[i] > 0.5 ? 70 : 0;
      img.data[i * 4] = 124; img.data[i * 4 + 1] = 58; img.data[i * 4 + 2] = 237; img.data[i * 4 + 3] = a;
    }
    tg.putImageData(img, 0, 0);
    g.drawImage(tint, 0, 0, c.width, c.height);
  }
  if (landmarks) {
    g.lineWidth = Math.max(2, c.width / 260);
    g.strokeStyle = 'rgba(196,181,253,.95)';
    g.beginPath();
    for (const [a, b] of BONES) {
      const p = landmarks[a], q = landmarks[b];
      g.moveTo(p.x * c.width, p.y * c.height); g.lineTo(q.x * c.width, q.y * c.height);
    }
    g.stroke();
    g.fillStyle = '#fff';
    for (const i of [0, 11, 12, 23, 24, 27, 28]) {
      const p = landmarks[i];
      g.beginPath(); g.arc(p.x * c.width, p.y * c.height, g.lineWidth * 1.6, 0, Math.PI * 2); g.fill();
    }
  }
  c.classList.toggle('mirror', S.mirror && S.rotate === 0);
}

/** Keep a silhouette (no photo) with the three measuring lines for the result screen. */
function snapshotSilhouette(mask, w, h, segs) {
  if (!mask) return null;
  const c = document.createElement('canvas');
  c.width = mask.width; c.height = mask.height;
  const g = c.getContext('2d');
  const img = g.createImageData(mask.width, mask.height);
  for (let i = 0; i < mask.data.length; i++) {
    const on = mask.data[i] > 0.5;
    img.data[i * 4] = 124; img.data[i * 4 + 1] = 58; img.data[i * 4 + 2] = 237; img.data[i * 4 + 3] = on ? 255 : 0;
  }
  g.putImageData(img, 0, 0);
  return { canvas: c, segs: segs || [] };
}

function paintSilhouette(target, vis) {
  if (!vis) { target.hidden = true; return; }
  target.hidden = false;
  const src = vis.canvas;
  // crop to the body
  const g0 = src.getContext('2d').getImageData(0, 0, src.width, src.height).data;
  let x0 = src.width, x1 = 0, y0 = src.height, y1 = 0;
  for (let y = 0; y < src.height; y++) for (let x = 0; x < src.width; x++) {
    if (g0[(y * src.width + x) * 4 + 3]) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  if (x1 <= x0) { target.hidden = true; return; }
  const pad = Math.round((y1 - y0) * 0.04);
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); x1 = Math.min(src.width, x1 + pad); y1 = Math.min(src.height, y1 + pad);
  const H = 520, k = H / (y1 - y0);
  target.width = Math.round((x1 - x0) * k); target.height = H;
  const g = target.getContext('2d');
  g.clearRect(0, 0, target.width, target.height);
  g.drawImage(src, x0, y0, x1 - x0, y1 - y0, 0, 0, target.width, target.height);
  // the measured widths: a bar across the body at each line, with end ticks
  const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink').trim() || '#fff';
  g.strokeStyle = ink; g.lineWidth = 2;
  for (const sg of vis.segs) {
    const y = (sg.y * src.height - y0) * k, a = (sg.x0 * src.width - x0) * k, b = (sg.x1 * src.width - x0) * k;
    g.beginPath(); g.moveTo(a, y); g.lineTo(b, y); g.moveTo(a, y - 6); g.lineTo(a, y + 6); g.moveTo(b, y - 6); g.lineTo(b, y + 6); g.stroke();
  }
}

// ── result ─────────────────────────────────────────────────────────────────
function renderResult({ m, frontVis, sideVis, note }) {
  const cats = [['tops', 'tops'], ['bottoms', 'bottoms'], ['one-pieces', 'onepiece']];
  $('sizes').innerHTML = '';
  let conf = 'none';
  for (const [cat, key] of cats) {
    const r = recommendSize(m, cat);
    if (cat === 'tops') conf = r.confidence;
    const li = document.createElement('li');
    li.innerHTML = `<span class="k"></span><span class="big"></span><span class="alt"></span>`;
    li.querySelector('.k').textContent = t(key);
    li.querySelector('.big').textContent = r.size || '–';
    li.querySelector('.alt').textContent = r.alternative ? t('orAlt')(r.alternative) : '';
    $('sizes').append(li);
  }
  const rows = [['chest', m.chest_cm], ['waist', m.waist_cm], ['hips', m.hip_cm], ['shoulders', m.shoulder_cm]];
  $('measures').innerHTML = '';
  for (const [k, v] of rows) {
    const div = document.createElement('div');
    div.innerHTML = '<dt></dt><dd><span class="num"></span><small></small></dd>';
    div.querySelector('dt').textContent = t(k);
    div.querySelector('.num').textContent = Math.round(v);
    div.querySelector('small').textContent = ' ' + t('cm');
    $('measures').append(div);
  }
  const qual = m.quality === 'high' ? 'high' : m.quality === 'low' ? 'low' : 'medium';
  $('conf').textContent = t('conf')[conf === 'none' ? 'none' : (qual === 'low' ? 'low' : conf)];
  $('conf').dataset.level = qual === 'low' ? 'low' : conf;
  $('resultNote').textContent = (note || '') + t('note');
  paintSilhouette($('silFront'), frontVis);
  paintSilhouette($('silSide'), sideVis);
}

// ── two-photo mode ─────────────────────────────────────────────────────────
const photos = { front: null, side: null };
function pickPhoto(which, file) {
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    photos[which] = img;
    const slot = $(which === 'front' ? 'slotFront' : 'slotSide');
    slot.style.backgroundImage = `url("${img.src}")`;
    slot.classList.add('has');
    $('measureBtn').disabled = !photos.front;
  };
  img.src = URL.createObjectURL(file);
}
function imageToCanvas(img) {
  const c = document.createElement('canvas');
  const k = Math.min(1, 1280 / Math.max(img.naturalWidth, img.naturalHeight));
  c.width = mult4(img.naturalWidth * k); c.height = mult4(img.naturalHeight * k);
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  return c;
}
function photoMsg(text, kind = '') { const el = $('photoMsg'); el.textContent = text; el.dataset.kind = kind; }
async function measurePhotos() {
  photoMsg(t('measuring'));
  $('measureBtn').disabled = true;
  try { await loadModel('IMAGE'); } catch (e) { console.error(e); photoMsg(t('modelError'), 'err'); $('measureBtn').disabled = false; return; }
  try { measurePhotosNow(); } finally { $('measureBtn').disabled = !photos.front; }
}
function measurePhotosNow() {
  const fc = imageToCanvas(photos.front);
  const f = detect(fc);
  const fchk = checkPose(f.landmarks, 'front', fc.width, fc.height);
  if (!fchk.ok) { photoMsg(t('photoFrontBad') + t(fchk.msg), 'err'); return; }
  const fw = frontWidths(f.landmarks, f.mask, fc.width, fc.height, S.height);
  let sd = null, sideVis = null, note = '';
  if (photos.side) {
    const sc = imageToCanvas(photos.side);
    const s = detect(sc);
    const schk = checkPose(s.landmarks, 'side', sc.width, sc.height);
    if (schk.ok) {
      sd = sideDepths(s.landmarks, s.mask, sc.width, sc.height, S.height);
      if (sd) sideVis = snapshotSilhouette(s.mask, sc.width, sc.height, sd.segs);
    }
    if (!sd) note = t('photoSideBad');
  }
  const m = combine([fw], sd ? [sd] : [], S.height);
  S.lastResult = { m, frontVis: snapshotSilhouette(f.mask, fc.width, fc.height, fw.segs), sideVis, note };
  photoMsg('');
  renderResult(S.lastResult);
  show('result');
}

// ── wiring ─────────────────────────────────────────────────────────────────
function init() {
  let lang = 'en', height = 175, theme = null;
  try {
    lang = localStorage.getItem('ss.lang') || (navigator.language || 'en').slice(0, 2);
    height = Number(localStorage.getItem('ss.height')) || 175;
    theme = localStorage.getItem('ss.theme');
  } catch {}
  const q = new URLSearchParams(location.search);
  if (q.get('lang')) lang = q.get('lang');
  if (q.get('theme')) theme = q.get('theme');
  setTheme(theme === 'light' || theme === 'dark' ? theme : null);
  setLang(lang);
  setHeight(q.get('h') || height);

  document.querySelectorAll('.langs button').forEach((b) => b.addEventListener('click', () => setLang(b.dataset.l)));
  $('themeBtn').addEventListener('click', () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark'));
  $('hMinus').addEventListener('click', () => setHeight(S.height - 1));
  $('hPlus').addEventListener('click', () => setHeight(S.height + 1));
  $('heightVal').addEventListener('change', (e) => setHeight(e.target.value));
  $('startCam').addEventListener('click', startScan);
  $('usePhotos').addEventListener('click', () => show('photos'));
  $('cancelScan').addEventListener('click', () => { stopScan(); show('start'); });
  $('skipSide').addEventListener('click', () => finish(true));
  $('rotateBtn').addEventListener('click', () => { S.rotateTouched = true; S.rotate = (S.rotate + 90) % 360; S.stable = 0; });
  $('camSel').addEventListener('change', async (e) => { try { S.rotateTouched = false; S.rotate = 0; await startCamera(e.target.value); } catch { status(t('camError'), 'err'); } });
  $('fileFront').addEventListener('change', (e) => pickPhoto('front', e.target.files[0]));
  $('fileSide').addEventListener('change', (e) => pickPhoto('side', e.target.files[0]));
  $('measureBtn').addEventListener('click', measurePhotos);
  $('photosBack').addEventListener('click', () => show('start'));
  $('again').addEventListener('click', () => { S.lastResult = null; show('start'); });
  if (!navigator.mediaDevices?.getUserMedia) $('startCam').disabled = true;
  document.addEventListener('visibilitychange', () => { if (document.hidden && S.running) { stopScan(); show('start'); } });
}
init();

// exposed for automated tests
window.__sizeScan = { S, measurePhotos, pickPhoto, photos };

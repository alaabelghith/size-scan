r"""
Fashion Mirror — Size Estimator (Phase A, per size_detection_plan.md)

Guided front + profile capture -> MediaPipe Pose QC -> geometric
measurement estimation -> hardcoded demo size-chart lookup.

STATUS: demo-grade. Not validated against tape-measure ground truth yet.
2026-10-11: front-view widths now measure the torso only (arms and hands that
touch the body are cut at the arm's inner edge); before, the silhouette's
first-to-last pixel included both arms and inflated chest/waist/hips.
Benchmark on real people before trusting this for a paying client — see
"Sell-side note" in 6_strategy/Fashion Mirror/size_detection_plan.md.
Always keep a Skip path in the kiosk flow; try-on must never depend on this.

Why geometric, not a learned model: no credible free/local model maps a
single photo -> absolute garment size (677-participant study: 9.15%
consistency). This module produces circumference *estimates* from body
landmarks + a segmentation silhouette, then buckets them into a size
chart with a confidence flag — same approach as the MIT-licensed
Live-Measurements-Api reference the plan cites, adapted here.

Setup (once):
    .venv\Scripts\python.exe -m pip install "mediapipe==0.10.21"
    (pinned: mediapipe 0.10.31+ removed the legacy mp.solutions.pose API
    this module depends on — see github.com/google-ai-edge/mediapipe/issues/6204.
    run_app.bat checks/installs this automatically.)

Standalone test (no camera, pure math — safe to run anywhere):
    .venv\Scripts\python.exe size_estimator.py --selftest

Live guided capture (needs webcam, run on the kiosk machine):
    .venv\Scripts\python.exe size_estimator.py --webcam

Importable (used by app.py):
    from size_estimator import SizeEstimator, recommend_size
    est = SizeEstimator()                     # loads MediaPipe Pose once
    if est.available:
        ok, msg = est.check_pose(frame_bgr, view="front")
        ...
        measurements = est.compute(height_cm=170)
        result = recommend_size(measurements, category="tops")
"""

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── MediaPipe Pose landmark indices (33-point model) ────────────────────────
NOSE, LEFT_EYE, RIGHT_EYE = 0, 2, 5
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 13, 14, 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

VIS_MIN = 0.55  # landmark visibility threshold to trust a point

# ── Anthropometric constants (rough averages — demo-grade) ──────────────────
# Eyes sit ~7% of total standing height below the top of the head.
EYE_TO_TOP_FRACTION = 0.07
# 2026-11-19 scale fix: the ankle landmark is NOT the floor. Eye height ~93.6% of
# stature, ankle (malleolus) ~3.9% (ANSUR-type averages) -> eye-to-ankle = 89.7%
# of stature. Before, 93% was used, which inflated every cm-per-px by ~3.7%.
EYE_HEIGHT_FRAC = 0.936
ANKLE_HEIGHT_FRAC = 0.039
# Cross-sections are rounder-rectangular than an ellipse: a pure ellipse under-reads
# the chest/hips. Shape factors on top of Ramanujan (tune with sizing.calibration).
SHAPE_FACTOR = {"chest": 1.06, "waist": 1.02, "hip": 1.05}
# Plausible depth/width ratios (profile depth vs front width) — outside = bad profile
# capture (arm in front, half-turned), clamped to the range.
DEPTH_RATIO = {"chest": (0.58, 0.85), "waist": (0.60, 0.95), "hip": (0.62, 0.88)}
DEFAULT_DEPTH_RATIO = {"chest": 0.70, "waist": 0.75, "hip": 0.72}
# Plausible circumference / stature ratios for adults; outside -> clamp + low confidence.
STATURE_BOUNDS = {"chest": (0.45, 0.78), "waist": (0.36, 0.72), "hip": (0.47, 0.80)}
# A-pose check: wrists must be at least this far (cm) outside the hip edge.
ARM_GAP_CM = 10.0
# Vertical position of chest/waist lines as a fraction of the
# shoulder->hip span (0 = shoulder line, 1 = hip line).
CHEST_FRAC = 0.18
WAIST_FRAC = 0.62

MIN_MASK_ROW_PIXELS = 8   # below this, treat a mask row as unreliable
# Front view: arms touching the body are cut off at their inner edge
# (arm centre line from the elbow/wrist landmarks, moved in by half an arm).
UPPER_ARM_HALF_CM = 4.5
FOREARM_HALF_CM = 3.5
MAX_GAP_FRAC = 0.004      # background gap (fraction of width) still counted as body


# ── Hardcoded demo size chart (Phase A — per plan, config-driven in Phase C) ─
SIZE_CHARTS = {
    "tops": {  # chest circumference, cm
        "S":  (82, 90),
        "M":  (90, 98),
        "L":  (98, 106),
        "XL": (106, 116),
        "XXL": (116, 128),
    },
    "bottoms": {  # waist primary, hip as tie-breaker, cm
        "S":  {"waist": (66, 74),  "hip": (88, 96)},
        "M":  {"waist": (74, 82),  "hip": (96, 104)},
        "L":  {"waist": (82, 90),  "hip": (104, 112)},
        "XL": {"waist": (90, 100), "hip": (112, 122)},
        "XXL": {"waist": (100, 114), "hip": (122, 138)},
    },
    "one-pieces": {  # sized on the larger of chest-band / hip-band
        "S":  {"chest": (82, 90),  "hip": (88, 96)},
        "M":  {"chest": (90, 98),  "hip": (96, 104)},
        "L":  {"chest": (98, 106), "hip": (104, 112)},
        "XL": {"chest": (106, 116), "hip": (112, 122)},
        "XXL": {"chest": (116, 128), "hip": (122, 138)},
    },
}
SIZE_ORDER = ["S", "M", "L", "XL", "XXL"]
GENDER_SIZE_CHARTS = {
    "women": SIZE_CHARTS,
    "men": {
        # 2026-11-20: Primark men's tops chart (chest cm): S 91-96, M 97-102, L 103-108, XL 109-118, XXL 119-124
        "tops": {"S": (90, 97), "M": (97, 103), "L": (103, 109), "XL": (109, 119), "XXL": (119, 128)},
        "bottoms": {"S": {"waist": (74, 82), "hip": (90, 98)}, "M": {"waist": (82, 90), "hip": (98, 106)}, "L": {"waist": (90, 98), "hip": (106, 114)}, "XL": {"waist": (98, 108), "hip": (114, 124)}, "XXL": {"waist": (108, 120), "hip": (124, 138)}},
        "one-pieces": {"S": {"chest": (88, 96), "hip": (90, 98)}, "M": {"chest": (96, 104), "hip": (98, 106)}, "L": {"chest": (104, 112), "hip": (106, 114)}, "XL": {"chest": (112, 120), "hip": (114, 124)}, "XXL": {"chest": (120, 130), "hip": (124, 138)}},
    },
}

# Approximate reference-chart waist ranges in cm, expressed as EU garment
# labels. Brands vary; these are guidance labels, not a universal fit guarantee.
EU_WOMENS_TROUSER_WAIST = {
    32: (60, 64), 34: (64, 68), 36: (68, 72),
    38: (72, 76), 40: (76, 80), 42: (80, 84), 44: (84, 88),
    46: (88, 92), 48: (92, 98),
}
EU_MENS_TROUSER_WAIST = {      # 2026-11-20: FR trouser labels (taille ~ waist/2; Celio 40=80, 44=88, 46=92 cm)
    36: (70, 74), 38: (74, 78), 40: (78, 82), 42: (82, 86), 44: (86, 90),
    46: (90, 94), 48: (94, 98), 50: (98, 102), 52: (102, 106), 54: (106, 112),
}


RETAIL_MAX = 48     # Tunisian mass retail labels stop at 48 (Alaa, 2026-11-20); bigger -> "48+"


def _pick_numeric_size(value: Optional[float], ranges: dict):
    if value is None:
        return None
    n = min(ranges, key=lambda size: _range_score(value, *ranges[size]))
    return n if n <= RETAIL_MAX else f"{RETAIL_MAX}+"


def recommend_size_set(measurements: dict) -> dict:
    """Return alpha and numeric reference sizes for both gender charts."""
    output = {}
    for gender in ("women", "men"):
        top = recommend_size(measurements, "tops", gender)
        bottom = recommend_size(measurements, "bottoms", gender)
        chart = EU_WOMENS_TROUSER_WAIST if gender == "women" else EU_MENS_TROUSER_WAIST
        output[gender] = {
            "tops": top.get("size"), "bottoms_alpha": bottom.get("size"),
            "trousers_eu": _pick_numeric_size(measurements.get("waist_cm"), chart),
        }
    return output


# ── Pure math helpers (no MediaPipe dependency — unit-testable) ─────────────
def ellipse_circumference_cm(width_cm: float, depth_cm: float) -> float:
    """Ramanujan's 2nd approximation for an ellipse with semi-axes
    a=width/2, b=depth/2. Treats a body cross-section as an ellipse —
    a coarse but standard approximation for this kind of estimator."""
    a, b = width_cm / 2.0, depth_cm / 2.0
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def row_run_px(mask: np.ndarray, y_row: int, band: int = 2, cx: Optional[float] = None,
               lo: float = -math.inf, hi: float = math.inf) -> Optional[tuple]:
    """(width, left, right) in px of the body on one mask row, median over a
    small band of rows. With `cx` (body centre line) only the run of foreground
    pixels containing the centre counts, so arms held away from the body are
    left out (gaps up to MAX_GAP_FRAC of the width are bridged); `lo`/`hi` cap
    the run where an arm touches the body. Without `cx`: first-to-last
    foreground pixel. None if the row has no reliable foreground.
    Same algorithm as src/estimator.js in the open-source Size Scan."""
    h, w = mask.shape[:2]
    y0, y1 = max(0, int(round(y_row)) - band), min(h, int(round(y_row)) + band + 1)
    if y0 >= y1:
        return None
    max_gap = max(1, int(round(w * MAX_GAP_FRAC)))
    ws, ls, rs = [], [], []
    for y in range(y0, y1):
        row = mask[y] > 0.5
        cols = np.where(row)[0]
        if cols.size < MIN_MASK_ROW_PIXELS:
            continue
        if cx is None:
            ws.append(int(cols[-1] - cols[0])); ls.append(int(cols[0])); rs.append(int(cols[-1]))
            continue
        c = int(round(min(w - 1, max(0, cx))))
        if not row[c]:
            near = cols[np.abs(cols - c) <= max_gap]
            if near.size == 0:
                continue
            c = int(near[np.argmin(np.abs(near - c))])
        l = r = c
        gap = 0
        for x in range(c - 1, -1, -1):
            if row[x]:
                l, gap = x, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
        gap = 0
        for x in range(c + 1, w):
            if row[x]:
                r, gap = x, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
        l, r = max(l, lo), min(r, hi)
        if r - l + 1 >= MIN_MASK_ROW_PIXELS:
            ws.append(r - l); ls.append(l); rs.append(r)
    if not ws:
        return None
    return float(np.median(ws)), float(np.median(ls)), float(np.median(rs))


def row_width_px(mask: np.ndarray, y_row: int, band: int = 2, cx: Optional[float] = None,
                 lo: float = -math.inf, hi: float = math.inf) -> Optional[float]:
    """Width in px of the body at a given row (see row_run_px)."""
    run = row_run_px(mask, y_row, band, cx, lo, hi)
    return run[0] if run else None


def _arm_limits(lm: dict, y: float, centre: float, scale: float) -> tuple:
    """(lo, hi) x-limits at row y from the inner edges of both arms, front view."""
    lo, hi = -math.inf, math.inf
    for ids in ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)):
        pts = [lm.get(i) for i in ids]
        if any(p is None or p[2] < 0.3 for p in pts):
            continue
        # 2026-11-20: the hand hangs ~0.8 forearm below the wrist landmark -> extend the forearm
        # segment, otherwise hands at hip level were counted as body (hip width +12 cm in the logs)
        (ex, ey, _), (wx, wy, wv) = pts[1], pts[2]
        pts = [pts[0], pts[1], (wx + 0.8 * (wx - ex), wy + 0.8 * (wy - ey), wv)]
        for k in range(2):
            (ax, ay, _), (bx, by, _) = pts[k], pts[k + 1]
            if min(ay, by) <= y <= max(ay, by) and by != ay:
                x = ax + (y - ay) / (by - ay) * (bx - ax)
                half = (UPPER_ARM_HALF_CM if k == 0 else FOREARM_HALF_CM) / scale
                if x < centre:
                    lo = max(lo, x + half)
                else:
                    hi = min(hi, x - half)
                break
    return lo, hi


def _range_score(value: float, lo: float, hi: float) -> float:
    """0 = dead center of the range, grows toward the edges / outside."""
    mid = (lo + hi) / 2.0
    half = max(0.5, (hi - lo) / 2.0)
    return abs(value - mid) / half


def _pick_from_ranges(value: float, ranges: dict) -> tuple:
    """Given {size: (lo, hi)}, return (best_size, neighbor_size_or_None,
    distance_to_boundary_cm)."""
    best, best_score = None, None
    for size, (lo, hi) in ranges.items():
        score = _range_score(value, lo, hi)
        if best_score is None or score < best_score:
            best, best_score = size, score

    order = [s for s in SIZE_ORDER if s in ranges]
    idx = order.index(best)
    lo, hi = ranges[best]
    dist_to_edge = min(value - lo, hi - value)
    neighbor = None
    if dist_to_edge < 2.0:  # within 2cm of a boundary -> flag the neighbor
        if value < (lo + hi) / 2.0 and idx > 0:
            neighbor = order[idx - 1]
        elif idx < len(order) - 1:
            neighbor = order[idx + 1]
    return best, neighbor, dist_to_edge


def recommend_size(measurements: dict, category: str, gender: str = "women") -> dict:
    """measurements: dict with any of chest_cm / waist_cm / hip_cm / quality.
    Returns {size, alternative, confidence, reason, basis}."""
    charts = GENDER_SIZE_CHARTS.get(gender, SIZE_CHARTS)
    chart = charts.get(category, charts["tops"])
    quality = measurements.get("quality", "medium")

    if category == "bottoms":
        waist = measurements.get("waist_cm")
        if waist is None:
            return _no_result("no waist measurement")
        ranges = {s: v["waist"] for s, v in chart.items()}
        size, neighbor, edge_dist = _pick_from_ranges(waist, ranges)
        basis = f"waist {waist:.0f}cm"
    elif category == "one-pieces":
        chest = measurements.get("chest_cm")
        hip = measurements.get("hip_cm")
        if chest is None or hip is None:
            return _no_result("no chest/hip measurement")
        chest_ranges = {s: v["chest"] for s, v in chart.items()}
        hip_ranges = {s: v["hip"] for s, v in chart.items()}
        size_c, _, edge_c = _pick_from_ranges(chest, chest_ranges)
        size_h, _, edge_h = _pick_from_ranges(hip, hip_ranges)
        # go with whichever band is larger — safer default for dresses/jumpsuits
        size = size_c if SIZE_ORDER.index(size_c) >= SIZE_ORDER.index(size_h) else size_h
        neighbor = size_h if size == size_c else size_c
        neighbor = neighbor if neighbor != size else None
        edge_dist = min(edge_c, edge_h)
        basis = f"chest {chest:.0f}cm / hip {hip:.0f}cm"
    else:  # tops
        chest = measurements.get("chest_cm")
        if chest is None:
            return _no_result("no chest measurement")
        size, neighbor, edge_dist = _pick_from_ranges(chest, chart)
        basis = f"chest {chest:.0f}cm"

    confidence = "high" if quality == "high" and edge_dist > 2.5 else \
                 "low" if quality == "low" or edge_dist < 1.0 else "medium"

    reason = f"Estimated from {basis}."
    if neighbor:
        reason += f" Close to the {neighbor} boundary — try {neighbor} if you prefer a different fit."

    return {
        "size": size,
        "alternative": neighbor,
        "confidence": confidence,
        "reason": reason,
        "basis": basis,
    }


def _no_result(why: str) -> dict:
    return {"size": None, "alternative": None, "confidence": "none",
            "reason": f"Could not estimate size ({why}).", "basis": None}


# ── MediaPipe-backed capture + QC (lazy import — never crash the kiosk) ─────
@dataclass
class _FrameData:
    landmarks: dict = field(default_factory=dict)   # idx -> (x_px, y_px, visibility)
    mask: Optional[np.ndarray] = None
    w: int = 0
    h: int = 0


class SizeEstimator:
    """Wraps MediaPipe Pose for the two-view guided capture used by
    app.py's size-capture screen. `available` is False (never raises) if
    mediapipe isn't installed or fails to load — callers must check it
    and skip straight to the try-on flow."""

    def __init__(self):
        self.available = False
        self._pose = None
        self.front: Optional[_FrameData] = None
        self.profile: Optional[_FrameData] = None
        self.front_frames: list = []
        self.profile_frames: list = []
        self.calib: dict = {}
        self.auto_height: dict = {}
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.available = True
        except Exception as e:
            print(f"! size_estimator: mediapipe unavailable ({e}) — sizing disabled")

    def close(self):
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass

    def _analyze(self, frame_rgb: np.ndarray) -> Optional[_FrameData]:
        if not self.available:
            return None
        h, w = frame_rgb.shape[:2]
        res = self._pose.process(frame_rgb)
        if not res.pose_landmarks:
            return None
        lm = {}
        for idx in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER,
                    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
                    LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE):
            p = res.pose_landmarks.landmark[idx]
            lm[idx] = (p.x * w, p.y * h, p.visibility)
        mask = res.segmentation_mask if res.segmentation_mask is not None else None
        return _FrameData(landmarks=lm, mask=mask, w=w, h=h)

    def check_pose(self, frame_rgb: np.ndarray, view: str) -> tuple:
        """Returns (ok, status_message). view = 'front' or 'profile'.
        Does NOT store the frame — call capture() once ok is True and
        stable for a few consecutive frames."""
        fd = self._analyze(frame_rgb)
        if fd is None:
            return False, "Step back so your whole body is visible"

        lm = fd.landmarks
        vis_ok = all(lm[i][2] > VIS_MIN for i in
                     (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP))
        ankles_ok = (lm[LEFT_ANKLE][2] > 0.3 or lm[RIGHT_ANKLE][2] > 0.3)
        if not ankles_ok:
            return False, "Step back — feet not visible"

        span = abs(lm[LEFT_ANKLE][1] - lm[NOSE][1])
        if span < fd.h * 0.45:
            return False, "Move back a little"
        if span > fd.h * 0.98:
            return False, "Move closer"

        shoulder_w = abs(lm[LEFT_SHOULDER][0] - lm[RIGHT_SHOULDER][0])

        if view == "front":
            if not vis_ok:
                return False, "Face the camera, arms slightly out"
            if shoulder_w < fd.w * 0.10:
                return False, "Face the camera directly"
            # arms must not touch the torso (biggest error source: hands at the waist)
            span_px = max(1.0, abs(lm[LEFT_ANKLE][1] - lm[NOSE][1]))
            cm_px = 160.0 / span_px            # rough, height-independent enough for a gate
            hip_l = min(lm[LEFT_HIP][0], lm[RIGHT_HIP][0])
            hip_r = max(lm[LEFT_HIP][0], lm[RIGHT_HIP][0])
            hip_half = (hip_r - hip_l) * 0.75     # hip joints sit inside the hip edge
            cxh = (hip_l + hip_r) / 2.0
            for wi in (LEFT_WRIST, RIGHT_WRIST):
                wx, wy, wv = lm[wi]
                if wv < 0.3:
                    return False, "Face the camera, arms slightly out"
                if abs(wx - cxh) - hip_half < ARM_GAP_CM / cm_px:
                    return False, "Face the camera, arms slightly out"
            return True, "Hold still..."
        else:  # profile
            if shoulder_w > fd.w * 0.12:      # 2026-11-20: 0.09 rejected big bodies already in profile
                return False, "Turn 90° to your side"
            if not (lm[LEFT_SHOULDER][2] > VIS_MIN or lm[RIGHT_SHOULDER][2] > VIS_MIN):
                return False, "Turn 90° to your side"
            return True, "Hold still..."

    def capture(self, frame_rgb: np.ndarray, view: str) -> bool:
        fd = self._analyze(frame_rgb)
        if fd is None:
            return False
        return self.add_frame(fd, view)

    def add_frame(self, fd: "_FrameData", view: str) -> bool:
        """Store an already-analysed frame (multi-frame median, 2026-11-19)."""
        if fd is None:
            return False
        lst = self.front_frames if view == "front" else self.profile_frames
        lst.append(fd)
        del lst[:-24]
        if view == "front":
            self.front = fd
        else:
            self.profile = fd
        return True

    def reset(self):
        self.front = None
        self.profile = None
        self.front_frames = []
        self.profile_frames = []

    # ---- per-frame geometry ------------------------------------------------
    @staticmethod
    def _scale(lm, height_cm):
        eye_y = (lm[LEFT_EYE][1] + lm[RIGHT_EYE][1]) / 2.0
        ankle_y = (lm[LEFT_ANKLE][1] + lm[RIGHT_ANKLE][1]) / 2.0
        return height_cm * (EYE_HEIGHT_FRAC - ANKLE_HEIGHT_FRAC) / max(1.0, ankle_y - eye_y)

    @staticmethod
    def _lines(lm):
        sy = (lm[LEFT_SHOULDER][1] + lm[RIGHT_SHOULDER][1]) / 2.0
        hy = (lm[LEFT_HIP][1] + lm[RIGHT_HIP][1]) / 2.0
        sx = (lm[LEFT_SHOULDER][0] + lm[RIGHT_SHOULDER][0]) / 2.0
        hx = (lm[LEFT_HIP][0] + lm[RIGHT_HIP][0]) / 2.0
        ys = {"chest": sy + CHEST_FRAC * (hy - sy), "waist": sy + WAIST_FRAC * (hy - sy), "hip": hy}
        xs = {"chest": sx + CHEST_FRAC * (hx - sx), "waist": sx + WAIST_FRAC * (hx - sx), "hip": hx}
        return ys, xs

    def _front_widths(self, f, height_cm):
        lm = f.landmarks
        scale = self._scale(lm, height_cm)
        ys, xs = self._lines(lm)
        out = {"shoulder": abs(lm[LEFT_SHOULDER][0] - lm[RIGHT_SHOULDER][0]) * scale}
        if f.mask is None:
            return None
        sy = ys["chest"] - CHEST_FRAC * (ys["hip"] - ys["chest"]) / (1 - CHEST_FRAC)   # shoulder line
        sh = ys["hip"] - sy
        def w_at(y, frac):
            c = xs["chest"] + (xs["hip"] - xs["chest"]) * max(0.0, (y - ys["chest"]) / max(1.0, ys["hip"] - ys["chest"]))
            lo, hi = _arm_limits(lm, y, c, scale)
            return row_width_px(f.mask, y, band=1, cx=c, lo=lo, hi=hi)
        def band(a, b, n=9):
            vals = [w_at(sy + sh * (a + (b - a) * i / (n - 1)), 0) for i in range(n)]
            return [v for v in vals if v]
        # 2026-11-20: bands instead of single rows. Natural waist = NARROWEST row (arms / folds only
        # ever add width); chest = median near the chest line; hip = low percentile around the hip line.
        ch, wa, hp = band(0.12, 0.26), band(0.45, 0.80, 13), band(0.92, 1.08)
        if not (ch and wa and hp):
            return None
        out["chest"] = float(np.median(ch)) * scale
        out["waist"] = float(min(wa)) * scale
        out["hip"] = float(np.percentile(hp, 30)) * scale
        # auto height (2026-11-20): head top = first mask row above the nose column, floor = ankle + 3.9 %
        nx = int(min(f.w - 1, max(0, lm[NOSE][0])))
        col = f.mask[:, max(0, nx - 3):nx + 4].max(axis=1) > 0.5
        rows = np.where(col)[0]
        ankle_y = (lm[LEFT_ANKLE][1] + lm[RIGHT_ANKLE][1]) / 2.0
        if rows.size:
            top = float(rows[0])
            out["px_h"] = (ankle_y - top) / (1 - ANKLE_HEIGHT_FRAC)
            out["feet_y"] = ankle_y / f.h
        return out

    def _profile_depths(self, p, height_cm):
        if p.mask is None:
            return None
        plm = p.landmarks
        scale = self._scale(plm, height_cm)
        ys, xs = self._lines(plm)
        out = {}
        sy = ys["chest"] - CHEST_FRAC * (ys["hip"] - ys["chest"]) / (1 - CHEST_FRAC)
        sh = ys["hip"] - sy
        def band(a, b, n=9):
            vals = []
            for i in range(n):
                fr = a + (b - a) * i / (n - 1); y = sy + sh * fr
                c = xs["chest"] + (xs["hip"] - xs["chest"]) * max(0.0, (y - ys["chest"]) / max(1.0, ys["hip"] - ys["chest"]))
                v = row_width_px(p.mask, y, band=1, cx=c)
                if v: vals.append(v)
            return vals
        ch, wa, hp = band(0.12, 0.26), band(0.45, 0.80, 13), band(0.92, 1.08)
        if not (ch and wa and hp):
            return None
        out["chest"] = float(np.median(ch)) * scale
        out["waist"] = float(min(wa)) * scale          # hands in front of the belly only add depth
        out["hip"] = float(np.percentile(hp, 30)) * scale
        return out

    @staticmethod
    def _median_dict(rows):
        rows = [r for r in rows if r]
        if not rows:
            return None
        return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}

    def predict_height(self):
        """cm from the front frames, needs sizing.auto_height.cm_per_px (fixed camera, shopper on the
        floor marker). Corrected by how far the feet are from the calibration spot (feet_y)."""
        ah = getattr(self, "auto_height", None) or {}
        k = ah.get("cm_per_px")
        if not k or not self.front_frames:
            return None
        rows = [self._front_widths(f, 170.0) for f in self.front_frames]
        rows = [r for r in rows if r and r.get("px_h")]
        if not rows:
            return None
        px = float(np.median([r["px_h"] for r in rows]))
        fy = float(np.median([r["feet_y"] for r in rows]))
        # nearer the camera = feet lower in frame = person bigger in px -> shrink, and the reverse
        corr = 1.0 - float(ah.get("feet_k", 0.0)) * (fy - float(ah.get("feet_y", fy)))
        return round(px * k * corr, 1), px, fy

    def compute(self, height_cm: float) -> dict:
        """Combine the stored front (+ profile) captures into measurements.
        Median over every stored frame of each view. Returns {} without a front view."""
        fronts = self.front_frames or ([self.front] if self.front else [])
        if not fronts:
            return {}
        profs = self.profile_frames or ([self.profile] if self.profile else [])
        calib = getattr(self, "calib", None) or {}

        W = self._median_dict([self._front_widths(f, height_cm) for f in fronts])
        used_mask = W is not None
        if not used_mask:      # landmark-only fallback
            lm = fronts[-1].landmarks
            scale = self._scale(lm, height_cm)
            sw = abs(lm[LEFT_SHOULDER][0] - lm[RIGHT_SHOULDER][0]) * scale
            hw = abs(lm[LEFT_HIP][0] - lm[RIGHT_HIP][0]) * scale * 1.35
            W = {"shoulder": sw, "chest": sw * 0.86, "waist": sw * 0.74, "hip": hw}
        D = self._median_dict([self._profile_depths(p, height_cm) for p in profs]) if profs else None
        used_profile = D is not None

        flags = []
        circ, depth = {}, {}
        for k in ("chest", "waist", "hip"):
            lo, hi = DEPTH_RATIO[k]
            if used_profile:
                r = D[k] / max(1e-3, W[k])
                if r < lo or r > hi:
                    flags.append(f"{k}_depth_clamped({r:.2f})")
                r = min(hi, max(lo, r))
            else:
                r = DEFAULT_DEPTH_RATIO[k]
            depth[k] = W[k] * r
            c = ellipse_circumference_cm(W[k], depth[k]) * SHAPE_FACTOR[k]
            c *= float(calib.get(k, 1.0))
            blo, bhi = STATURE_BOUNDS[k]
            if c < blo * height_cm or c > bhi * height_cm:
                flags.append(f"{k}_out_of_range({c:.0f})")
                c = min(bhi * height_cm, max(blo * height_cm, c))
            circ[k] = c

        quality = "high" if (used_mask and used_profile and len(fronts) >= 4) else \
                  "medium" if (used_mask or used_profile) else "low"
        if any("out_of_range" in f for f in flags):
            quality = "low"
        elif flags and quality == "high":
            quality = "medium"

        return {
            "height_cm": height_cm,
            "shoulder_cm": round(W["shoulder"], 1),
            "chest_cm": round(circ["chest"], 1),
            "waist_cm": round(circ["waist"], 1),
            "hip_cm": round(circ["hip"], 1),
            "quality": quality,
            "raw": {"width": {k: round(v, 1) for k, v in W.items()},
                    "depth": {k: round(v, 1) for k, v in depth.items()},
                    "frames": [len(fronts), len(profs)], "flags": flags},
        }


# ── Self-test: pure math, no camera / no mediapipe required ─────────────────
def _selftest():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("── size_estimator selftest (synthetic data, no camera) ──")

    # 1) ellipse circumference sanity: a circle of radius 15cm -> 2*pi*15
    c = ellipse_circumference_cm(30, 30)
    expected = 2 * math.pi * 15
    print(f"circle check: {c:.2f} vs expected {expected:.2f}  "
          f"{'OK' if abs(c - expected) < 0.01 else 'FAIL'}")

    # 2) row_width_px on a synthetic silhouette mask
    mask = np.zeros((200, 100), dtype=np.float32)
    mask[50:150, 30:70] = 1.0  # 40px-wide torso band
    w = row_width_px(mask, 100)
    print(f"row_width_px: {w}px (expected ~39-40)  "
          f"{'OK' if w and 38 <= w <= 40 else 'FAIL'}")
    assert row_width_px(mask, 10) is None, "empty row should return None"
    print("row_width_px empty-row guard: OK")

    # 3) recommend_size across categories
    r1 = recommend_size({"chest_cm": 94, "quality": "high"}, "tops")
    print(f"tops chest=94cm -> {r1}")
    assert r1["size"] == "M"

    r2 = recommend_size({"waist_cm": 89, "quality": "medium"}, "bottoms")
    print(f"bottoms waist=89cm -> {r2}")
    assert r2["size"] == "L" and r2["alternative"] == "XL"

    r3 = recommend_size({"chest_cm": 100, "hip_cm": 118, "quality": "medium"}, "one-pieces")
    print(f"one-pieces chest=100/hip=118 -> {r3}")
    assert r3["size"] == "XL"  # driven by the larger hip band

    sizes = recommend_size_set({"chest_cm": 100, "waist_cm": 84, "hip_cm": 104, "quality": "high"})
    assert set(sizes) == {"men", "women"}
    assert sizes["women"]["trousers_eu"] in list(range(32, 49, 2)) + ["48+"]
    assert sizes["men"]["trousers_eu"] in list(range(36, 49, 2)) + ["48+"]
    assert sizes["women"]["tops"] in SIZE_ORDER and sizes["men"]["tops"] in SIZE_ORDER
    print(f"size crosswalk: {sizes}")

    # 4) full compute() pipeline with synthetic landmarks + masks
    #    (simulates a ~178cm person, no real mediapipe needed)
    est = SizeEstimator.__new__(SizeEstimator)  # bypass mediapipe import
    est.available = True
    est.front_frames, est.profile_frames, est.calib = [], [], {}
    # Proportions matter here: shoulder width should be ~22% of standing
    # height (real biacromial-breadth average), not an arbitrary guess —
    # otherwise the derived circumference comes out anatomically tiny and
    # this smoke test would pass math it shouldn't.
    est.front = _FrameData(
        landmarks={
            NOSE: (200, 120, 0.9), LEFT_EYE: (190, 110, 0.9), RIGHT_EYE: (210, 110, 0.9),
            LEFT_SHOULDER: (105, 220, 0.9), RIGHT_SHOULDER: (295, 220, 0.9),
            LEFT_HIP: (125, 480, 0.9), RIGHT_HIP: (275, 480, 0.9),
            LEFT_ANKLE: (170, 900, 0.9), RIGHT_ANKLE: (230, 900, 0.9),
        },
        mask=_make_body_mask(),
        w=400, h=960,
    )
    est.profile = None
    m = est.compute(height_cm=178)
    print(f"compute() front-only: {m}")
    assert m["quality"] in ("medium", "low", "high")
    assert 60 < m["chest_cm"] < 140

    print("── all selftest checks passed ──")


def _make_body_mask() -> np.ndarray:
    """Rough synthetic silhouette proportioned like an adult torso
    (~200px wide near the chest tapering to ~170px near the hips, against
    an ~850px total-height scale) for the compute() smoke test above."""
    mask = np.zeros((960, 400), dtype=np.float32)
    for y in range(200, 500):
        t = (y - 200) / 300.0
        half_w = int(100 - 15 * t)
        cx = 200
        mask[y, cx - half_w:cx + half_w] = 1.0
    return mask


# ── Live webcam CLI (manual test on the kiosk machine) ──────────────────────
def _webcam_cli():
    import cv2
    est = SizeEstimator()
    if not est.available:
        print("mediapipe not available — install it first: "
              ".venv\\Scripts\\python.exe -m pip install mediapipe")
        return

    cap = cv2.VideoCapture(0)
    stage = "front"
    stable = 0
    height_cm = 170
    print(f"Height set to {height_cm}cm (edit HEIGHT_CM in this script to change).")
    print("Stand back so your whole body is visible. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = frame[:, :, ::-1]
        ok, msg = est.check_pose(rgb, view=stage)
        stable = stable + 1 if ok else 0
        print(f"\r[{stage}] {msg} (stable={stable})   ", end="", flush=True)

        if stable >= 10:
            est.capture(rgb, view=stage)
            stable = 0
            if stage == "front":
                stage = "profile"
                print("\nFront captured. Turn 90 degrees to your side.")
            else:
                print("\nProfile captured. Computing...")
                break

        import cv2 as _cv2
        _cv2.imshow("size_estimator test", frame)
        if _cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    import cv2 as _cv2
    _cv2.destroyAllWindows()

    if est.front:
        measurements = est.compute(height_cm=height_cm)
        print("Measurements:", measurements)
        print("Tops size:", recommend_size(measurements, "tops"))
        print("Bottoms size:", recommend_size(measurements, "bottoms"))
    est.close()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--webcam" in sys.argv:
        _webcam_cli()
    else:
        print(__doc__)

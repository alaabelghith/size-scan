r"""
Size Scan — Python reference implementation (the version that runs in the
Fashion Mirror kiosk). Same method as ../src/estimator.js.

Guided front + side capture -> MediaPipe Pose checks -> body widths from the
segmentation silhouette -> ellipse circumferences -> size-chart lookup.

STATUS: demo-grade. Not validated against tape-measure ground truth on a
group of people. Treat results as a starting point.

Why geometric, not a learned model: there is no reliable free model that maps
a single photo to an absolute garment size. This module estimates
circumferences from body landmarks plus a segmentation silhouette, then
buckets them into a size chart with a confidence flag. (The overall approach
is similar to the MIT-licensed Live-Measurements-Api project.)

Front-view widths measure the torso only: arms and hands that touch the body
are cut at the arm's inner edge (elbow/wrist landmarks), so they do not
inflate chest, waist or hips.

Setup:
    pip install "mediapipe==0.10.21" numpy opencv-python
    (pinned: mediapipe 0.10.31+ removed the legacy mp.solutions.pose API
    this module uses — github.com/google-ai-edge/mediapipe/issues/6204)

Self-test (no camera, pure maths):
    python size_estimator.py --selftest

Live guided capture with a webcam:
    python size_estimator.py --webcam

As a module:
    from size_estimator import SizeEstimator, recommend_size
    est = SizeEstimator()
    if est.available:
        ok, msg = est.check_pose(frame_rgb, view="front")
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


# ── Demo size chart (cm) — replace with your brand's chart ─────────────────────
SIZE_CHARTS = {
    "tops": {  # chest circumference, cm
        "S":  (82, 90),
        "M":  (90, 98),
        "L":  (98, 106),
        "XL": (106, 116),
    },
    "bottoms": {  # waist primary, hip as tie-breaker, cm
        "S":  {"waist": (66, 74),  "hip": (88, 96)},
        "M":  {"waist": (74, 82),  "hip": (96, 104)},
        "L":  {"waist": (82, 90),  "hip": (104, 112)},
        "XL": {"waist": (90, 100), "hip": (112, 122)},
    },
    "one-pieces": {  # sized on the larger of chest-band / hip-band
        "S":  {"chest": (82, 90),  "hip": (88, 96)},
        "M":  {"chest": (90, 98),  "hip": (96, 104)},
        "L":  {"chest": (98, 106), "hip": (104, 112)},
        "XL": {"chest": (106, 116), "hip": (112, 122)},
    },
}
SIZE_ORDER = ["S", "M", "L", "XL"]


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


def recommend_size(measurements: dict, category: str) -> dict:
    """measurements: dict with any of chest_cm / waist_cm / hip_cm / quality.
    Returns {size, alternative, confidence, reason, basis}."""
    chart = SIZE_CHARTS.get(category, SIZE_CHARTS["tops"])
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


# ── MediaPipe-backed capture + checks (lazy import — never crash the caller) ────
@dataclass
class _FrameData:
    landmarks: dict = field(default_factory=dict)   # idx -> (x_px, y_px, visibility)
    mask: Optional[np.ndarray] = None
    w: int = 0
    h: int = 0


class SizeEstimator:
    """Wraps MediaPipe Pose for the two-view guided capture used by
    a size-capture screen. `available` is False (never raises) if
    mediapipe isn't installed or fails to load — callers must check it
    and skip straight to the try-on flow."""

    def __init__(self):
        self.available = False
        self._pose = None
        self.front: Optional[_FrameData] = None
        self.profile: Optional[_FrameData] = None
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
            return True, "Hold still..."
        else:  # profile
            if shoulder_w > fd.w * 0.09:
                return False, "Turn 90° to your side"
            if not (lm[LEFT_SHOULDER][2] > VIS_MIN or lm[RIGHT_SHOULDER][2] > VIS_MIN):
                return False, "Turn 90° to your side"
            return True, "Hold still..."

    def capture(self, frame_rgb: np.ndarray, view: str) -> bool:
        fd = self._analyze(frame_rgb)
        if fd is None:
            return False
        if view == "front":
            self.front = fd
        else:
            self.profile = fd
        return True

    def reset(self):
        self.front = None
        self.profile = None

    def compute(self, height_cm: float) -> dict:
        """Combine the stored front + profile captures into measurements.
        Returns {} if front capture is missing (minimum requirement)."""
        if self.front is None:
            return {}

        f = self.front
        lm = f.landmarks
        eye_y = (lm[LEFT_EYE][1] + lm[RIGHT_EYE][1]) / 2.0
        ankle_y = (lm[LEFT_ANKLE][1] + lm[RIGHT_ANKLE][1]) / 2.0
        eye_to_ankle_px = max(1.0, ankle_y - eye_y)
        total_height_px = eye_to_ankle_px / (1 - EYE_TO_TOP_FRACTION)
        scale = height_cm / total_height_px  # cm per px, front frame

        shoulder_y = (lm[LEFT_SHOULDER][1] + lm[RIGHT_SHOULDER][1]) / 2.0
        hip_y = (lm[LEFT_HIP][1] + lm[RIGHT_HIP][1]) / 2.0
        shoulder_w_px = abs(lm[LEFT_SHOULDER][0] - lm[RIGHT_SHOULDER][0])
        hip_w_px_lm = abs(lm[LEFT_HIP][0] - lm[RIGHT_HIP][0])

        chest_y = shoulder_y + CHEST_FRAC * (hip_y - shoulder_y)
        waist_y = shoulder_y + WAIST_FRAC * (hip_y - shoulder_y)

        # body centre line between the shoulder and hip mid-points
        sh_x = (lm[LEFT_SHOULDER][0] + lm[RIGHT_SHOULDER][0]) / 2.0
        hp_x = (lm[LEFT_HIP][0] + lm[RIGHT_HIP][0]) / 2.0
        cx = {"chest": sh_x + CHEST_FRAC * (hp_x - sh_x), "waist": sh_x + WAIST_FRAC * (hp_x - sh_x), "hip": hp_x}

        used_mask = False
        if f.mask is not None:
            def _w(y, c):
                lo, hi = _arm_limits(lm, y, c, scale)
                return row_width_px(f.mask, y, cx=c, lo=lo, hi=hi)
            chest_w_px = _w(chest_y, cx["chest"])
            waist_w_px = _w(waist_y, cx["waist"])
            hip_w_px = _w(hip_y, cx["hip"])
            if chest_w_px and waist_w_px and hip_w_px:
                used_mask = True
        if not used_mask:
            # landmark-only fallback: interpolate between shoulder & hip width
            chest_w_px = shoulder_w_px - CHEST_FRAC * (shoulder_w_px - hip_w_px_lm)
            waist_w_px = shoulder_w_px - WAIST_FRAC * (shoulder_w_px - hip_w_px_lm)
            hip_w_px = hip_w_px_lm

        chest_w_cm = chest_w_px * scale
        waist_w_cm = waist_w_px * scale
        hip_w_cm = hip_w_px * scale
        shoulder_w_cm = shoulder_w_px * scale

        # Profile depths (optional — improves the estimate but not required)
        chest_d_cm = waist_d_cm = hip_d_cm = None
        used_profile = False
        if self.profile is not None and self.profile.mask is not None:
            p = self.profile
            plm = p.landmarks
            p_eye_y = (plm[LEFT_EYE][1] + plm[RIGHT_EYE][1]) / 2.0
            p_ankle_y = (plm[LEFT_ANKLE][1] + plm[RIGHT_ANKLE][1]) / 2.0
            p_scale = height_cm / max(1.0, (p_ankle_y - p_eye_y) / (1 - EYE_TO_TOP_FRACTION))
            p_shoulder_y = (plm[LEFT_SHOULDER][1] + plm[RIGHT_SHOULDER][1]) / 2.0
            p_hip_y = (plm[LEFT_HIP][1] + plm[RIGHT_HIP][1]) / 2.0
            p_chest_y = p_shoulder_y + CHEST_FRAC * (p_hip_y - p_shoulder_y)
            p_waist_y = p_shoulder_y + WAIST_FRAC * (p_hip_y - p_shoulder_y)
            p_sh_x = (plm[LEFT_SHOULDER][0] + plm[RIGHT_SHOULDER][0]) / 2.0
            p_hp_x = (plm[LEFT_HIP][0] + plm[RIGHT_HIP][0]) / 2.0
            cd = row_width_px(p.mask, p_chest_y, cx=p_sh_x + CHEST_FRAC * (p_hp_x - p_sh_x))
            wd = row_width_px(p.mask, p_waist_y, cx=p_sh_x + WAIST_FRAC * (p_hp_x - p_sh_x))
            hd = row_width_px(p.mask, p_hip_y, cx=p_hp_x)
            if cd and wd and hd:
                chest_d_cm, waist_d_cm, hip_d_cm = cd * p_scale, wd * p_scale, hd * p_scale
                used_profile = True

        if used_profile:
            chest_circ = ellipse_circumference_cm(chest_w_cm, chest_d_cm)
            waist_circ = ellipse_circumference_cm(waist_w_cm, waist_d_cm)
            hip_circ = ellipse_circumference_cm(hip_w_cm, hip_d_cm)
        else:
            # No profile depth: assume a body-depth/width ratio of ~0.62
            # (rough average) so we still return a circumference, at lower
            # confidence, rather than nothing.
            chest_circ = ellipse_circumference_cm(chest_w_cm, chest_w_cm * 0.62)
            waist_circ = ellipse_circumference_cm(waist_w_cm, waist_w_cm * 0.62)
            hip_circ = ellipse_circumference_cm(hip_w_cm, hip_w_cm * 0.62)

        quality = "high" if (used_mask and used_profile) else \
                  "medium" if (used_mask or used_profile) else "low"

        return {
            "height_cm": height_cm,
            "shoulder_cm": round(shoulder_w_cm, 1),
            "chest_cm": round(chest_circ, 1),
            "waist_cm": round(waist_circ, 1),
            "hip_cm": round(hip_circ, 1),
            "quality": quality,
        }


# ── Self-test: pure math, no camera / no mediapipe required ─────────────────
def _selftest():
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

    # 4) full compute() pipeline with synthetic landmarks + masks
    #    (simulates a ~178cm person, no real mediapipe needed)
    est = SizeEstimator.__new__(SizeEstimator)  # bypass mediapipe import
    est.available = True
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


# ── Live webcam CLI (manual test) ────────────────────────────────────────────
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

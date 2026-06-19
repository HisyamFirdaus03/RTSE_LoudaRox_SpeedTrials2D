"""
agent_policy.py
===============
Decision logic for the SpeedTrials2D agent, kept completely separate from the
networking / real-time scheduler scaffold in `sample_drive.py`.

Design goal: the brain is a swappable `Policy`. Tonight we ship `HeuristicPolicy`
(pure OpenCV). Later you can drop in `LearnedPolicy` (a trained model) by changing
ONE line in sample_drive.py -- nothing else has to change.

    front_frame (BGR np.ndarray)  ->  Policy.act()  ->  (steering, acceleration)

    steering      : -1.0 (full left)  ..  +1.0 (full right)
    acceleration  : -1.0 (reverse)    ..  +1.0 (full forward)

All tunable numbers live in CONFIG below so you can calibrate fast on real frames
(see calibrate.py / the debug overlay) without hunting through the logic.
"""

import os
import json
import time
import cv2
import numpy as np

CALIBRATION_FILE = "color_calibration.json"


# =========================================================================
# CONFIG  -- tune these on real frames (HSV is OpenCV's: H 0-179, S/V 0-255)
# =========================================================================
CONFIG = {
    # ---- HSV colour ranges for the three token types -------------------
    # Each entry is a list of (lower, upper) HSV pairs. Red needs two ranges
    # because its hue wraps around 0/180.
    # Generous fixed bands: wide hue (the 3 token hues are far apart) + a
    # saturation/brightness floor so the dull grey road and dark scenery are
    # excluded. The S/V floor is what separates tokens from background, NOT a
    # tight hue. Catches the pale translucent orbs too.
    "green": [((32, 60, 60), (92, 255, 255))],
    "red":   [((0, 45, 60), (13, 255, 255)),        # low S floor -> catch pale/salmon reds
              ((163, 45, 60), (179, 255, 255))],
    "yellow":[((16, 70, 90), (35, 255, 255))],
    # EV2 police car: a BLUE car on the road. Detecting it = the police event is
    # live -> we must collect a RED within 5s (and not hit the car).
    "police": [((100, 70, 40), (130, 255, 255))],
    # Use the narrow auto-calibrated file instead of the generous bands above?
    "use_calibration_file": False,

    # ---- EV2 Police event ----------------------------------------------
    "enable_police_event":   True,   # master toggle (flip False if it misbehaves)
    "police_min_area_frac":  0.0040, # a police CAR is big -> require a sizable blue
                                     # blob so stray blue pixels can't false-trigger
    "police_min_cy_frac":    0.58,   # only count police in the NEAR half of the road;
                                     # a tiny blue speck at the horizon must NOT trigger
    "police_grab_secs":      5.0,    # seek/collect a red for this long after it appears
    "police_weight":         5.0,    # how hard to avoid the police car (collision = GAME OVER)
    "save_event_frames":     True,   # auto-save frames when police is detected -> police_frames/
                                     # (so you can inspect the car without fast screenshots)

    # ---- Region of interest (the drivable road) ------------------------
    # A trapezoid: narrow near the horizon, wide near the car. Fractions of
    # frame width/height. CRITICAL: the bottom edge must stop ABOVE the
    # player's own red car (which sits at the bottom-center of the front
    # camera) -- otherwise the car's red body reads as a permanent red hazard
    # and its orange lights as yellow. roi_bottom_y crops the car out.
    "roi_top_y":      0.45,   # horizon side
    "roi_bottom_y":   0.80,   # just above the player's car (was 0.98 = included it)
    "roi_top_half_w": 0.12,   # half-width at the top, fraction of W
    "roi_bot_half_w": 0.44,   # half-width at the bottom, fraction of W

    # ---- Detection -----------------------------------------------------
    "min_token_area_frac": 0.0006,  # ignore blobs smaller than this * (W*H)
    # After filling, tokens (even translucent ring-shaped ones) are round
    # disks; thin striped posts stay elongated. Kept LENIENT so real orbs are
    # never dropped -- proximity weighting + ROI handle the rest. Raise toward
    # 0.6 only if barriers clearly still pollute the masks.
    "min_circularity": 0.42,

    # ---- Steering ------------------------------------------------------
    "steer_gain":     2.2,    # P-gain on normalized horizontal error
    "dodge_steer":    1.0,    # magnitude used when escaping a hazard
    "smoothing":      0.5,    # low-pass alpha: s = a*new + (1-a)*prev (higher = snappier)
    "center_x_frac":  0.5,    # where the car sits horizontally (bottom-center)

    # ---- Potential-field path planner ----------------------------------
    # Instead of reacting to ONE token, we score many candidate lateral paths:
    # greens pull, hazards push (with a wider buffer), weighted by proximity.
    # This threads between tokens instead of flip-flopping (which averaged to 0).
    "n_candidates":     21,     # how many lateral target positions to score
    "cand_min_frac":    0.12,   # leftmost reachable target (fraction of W)
    "cand_max_frac":    0.88,   # rightmost reachable target
    "green_reward":     1.2,    # attraction strength of a green token
    "hazard_penalty":   3.6,    # repulsion strength of a red/yellow (> reward!)
    "collect_sigma_frac": 0.06, # how close to a green's x counts as "collecting"
    "avoid_sigma_frac":   0.11, # red no-go radius ~= car width. The car is ~1 lane
                                # wide, so a green hugging a red CAN'T be taken
                                # without clipping the red -> skip it. Avoiding red
                                # is prioritised over grabbing a risky green.
    "lane_cost":        0.3,    # penalty for steering far from center (lower = will take side lanes)
    "yellow_weight":    1.15,   # avoid yellow as hard as (or harder than) red: its
                                # random debuff -- especially "lose colour vision" --
                                # can blind a vision-based agent, so steer clear of it.

    # Hazard "panic" zone: a hazard this close AND near our center cuts throttle.
    "hazard_near_y_frac":  0.62,  # "close" = centroid below this fraction of H
    "hazard_path_half_w":  0.18,  # "in front" = |cx - center| < this * W

    # ---- Acceleration (adaptive throttle) ------------------------------
    # Full speed when the road ahead is clear; ease off SMOOTHLY as the nearest
    # in-path hazard gets closer, so the steering has time to dodge it. The car
    # is ~1 lane wide and can't change lanes instantly -- slowing in danger is
    # what actually prevents red hits (and a red also costs -20% speed, so this
    # helps distance too).
    "accel_cruise":   1.0,    # normal forward throttle (clear road)
    "accel_dodge":    0.40,   # throttle when a hazard is right on top of us

    # ---- Road surface gate (tokens only exist ON the grey asphalt) -----
    # The grass is green and pollutes the green mask; the only reliable way to
    # exclude it is to detect tokens ONLY where there's road. We build a grey-
    # asphalt mask, dilate it to cover tokens sitting on it, and AND every
    # colour mask with it. This kills grass-as-green and grass-edge clutter.
    "road_sat_max":   60,     # road is low-saturation grey: S below this (stricter = less grass)
    "road_val_min":   40,     # ...and not pitch black
    "road_dilate":    17,     # px to grow road mask; only needs to cover the thin
                              # colored token RIM (then _fill_blobs rebuilds the disk)

    # ---- Special events ------------------------------------------------
    # EV1 "Darkness": the whole screen goes dark and the rule is to brake fully
    # (accel = -1). We detect it by mean brightness (V channel) collapsing well
    # below a normal night scene. Tune this if it triggers in normal play (lower
    # it) or never triggers during the dark event (raise it).
    "darkness_v_mean": 45,

    # ---- Lane-keep fallback (no token actionable) ----------------------
    "lanekeep_gain":  0.6,    # gentle pull toward the road centroid

    # ---- Degraded-vision fallback (yellow "lose colour" debuff) --------
    # Mean saturation of the WHOLE frame; if it collapses, colour vision is
    # likely scrambled -> fall back to lane-keep. DISABLED by default (0) to
    # avoid false triggers -- enable (try ~25) only AFTER you've observed a real
    # debuffed frame (via the debug overlay or logged dataset) and know the value.
    "blind_sat_mean": 0,

    # ---- Tooling -------------------------------------------------------
    "debug": True,           # draw the live overlay window (tuning only)
    "log_data": False,        # save (frame, action) pairs for imitation learning
    "log_dir": "dataset",
}


def load_color_overrides(path=CALIBRATION_FILE):
    """Load green/red/yellow HSV ranges produced by auto_calibrate.py, if present.
    Returns {} when the file is missing so defaults in CONFIG are used."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        out = {}
        for k in ("green", "red", "yellow"):
            if k in data:
                out[k] = [(tuple(lo), tuple(hi)) for lo, hi in data[k]]
        print(f"[agent_policy] loaded colour calibration from {path}")
        return out
    except Exception as e:
        print(f"[agent_policy] could not load {path}: {e} -- using defaults")
        return {}


# =========================================================================
# Small CV helpers
# =========================================================================
def _build_color_mask(hsv, ranges):
    """OR together every (lower, upper) HSV band for one token colour, despeckled.
    (Filling is done later, AFTER restricting to the road, by _fill_blobs.)"""
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _fill_blobs(mask):
    """Fill each external contour so a translucent orb's ring (colored rim,
    washed-out center) becomes a solid disk -- otherwise the round token reads
    as a non-round ring and gets shape-rejected. Works for any hole size."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled


def _roi_polygon(w, h):
    """Trapezoid covering the road; returns an (N,2) int array of vertices."""
    cx = w * CONFIG["center_x_frac"]
    top_y = h * CONFIG["roi_top_y"]
    bot_y = h * CONFIG["roi_bottom_y"]
    thw = w * CONFIG["roi_top_half_w"]
    bhw = w * CONFIG["roi_bot_half_w"]
    return np.array([
        [cx - thw, top_y],
        [cx + thw, top_y],
        [cx + bhw, bot_y],
        [cx - bhw, bot_y],
    ], np.int32)


def _detect_tokens(mask, min_area, min_circularity=0.0):
    """
    Find round token blobs in a masked image.
    Returns a list of dicts: {cx, cy, area} in pixel coords. 'Closer to the
    car' == larger cy. Non-round blobs (barriers/signs) are rejected by the
    circularity gate: circ = 4*pi*area / perimeter^2  (1.0 == perfect circle).
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < min_circularity:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        out.append({
            "cx": M["m10"] / M["m00"],
            "cy": M["m01"] / M["m00"],
            "area": area,
            "circ": circ,
        })
    return out


def _closest(tokens):
    """Token nearest the car = the one lowest in the frame (max cy)."""
    return max(tokens, key=lambda t: t["cy"]) if tokens else None


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


# =========================================================================
# Policy interface
# =========================================================================
class Policy:
    """Swappable brain. Implement act(); return (steering, acceleration)."""

    def act(self, front_frame, back_frame=None):
        raise NotImplementedError

    def reset(self):
        """Clear any per-episode state (called between runs if desired)."""
        pass


# =========================================================================
# HeuristicPolicy -- tonight's submission (pure OpenCV)
# =========================================================================
class HeuristicPolicy(Policy):
    """
    Priority of behaviours, every frame:
        1. AVOID  a close red/yellow token in our path   (hard override)
        2. SEEK   the nearest reachable green token
        3. KEEP   the car centered on the road            (survival fallback)
    Output steering is low-pass filtered to avoid twitchy oscillation at the
    200 Hz control rate.
    """

    def __init__(self, config=None):
        self.cfg = dict(CONFIG)
        if self.cfg.get("use_calibration_file", False):
            self.cfg.update(load_color_overrides())   # narrow auto-calibrated colours
        if config:
            self.cfg.update(config)                   # explicit overrides win
        self._prev_steer = 0.0
        self._police_until = 0.0                              # EV2: seek-red deadline
        self._last_save = 0.0                                 # rate-limit police-frame saves
        self._logger = DataLogger(self.cfg["log_dir"]) if self.cfg["log_data"] else None

    def reset(self):
        self._prev_steer = 0.0
        self._police_until = 0.0

    # -- main entry ------------------------------------------------------
    def act(self, front_frame, back_frame=None):
        if front_frame is None:
            return 0.0, self.cfg["accel_cruise"]

        h, w = front_frame.shape[:2]
        cfg = self.cfg
        center_x = w * cfg["center_x_frac"]
        min_area = cfg["min_token_area_frac"] * (w * h)

        # ROI mask (the road trapezoid).
        roi_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi_mask, [_roi_polygon(w, h)], 255)

        hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)

        # --- EV1 "Darkness" event: screen goes dark -> brake fully ------
        # The event rule is to fully decelerate; not braking risks a penalty.
        bright = float(hsv[:, :, 2].mean())
        if bright < cfg["darkness_v_mean"]:
            self._maybe_debug(front_frame, roi_mask, {}, 0.0, f"EV1 DARK({bright:.0f})->BRAKE")
            return 0.0, -1.0

        # --- Degraded-vision check (yellow "lose colour" debuff) --------
        if cfg["blind_sat_mean"] > 0 and cv2.mean(hsv[:, :, 1])[0] < cfg["blind_sat_mean"]:
            steer = self._lane_keep(hsv, roi_mask, w, center_x)
            steer = self._smooth(steer)
            self._maybe_debug(front_frame, roi_mask, {}, steer, "BLIND->lanekeep")
            return steer, cfg["accel_cruise"]

        # --- Detect tokens ONLY on the road surface (excludes grass) ----
        road_region = self._road_region(hsv, roi_mask)

        def cmask(ranges):
            m = cv2.bitwise_and(_build_color_mask(hsv, ranges), road_region)
            return _fill_blobs(m)

        green_m  = cmask(cfg["green"])
        red_m    = cmask(cfg["red"])
        yellow_m = cmask(cfg["yellow"])

        circ = cfg["min_circularity"]
        greens = _detect_tokens(green_m,  min_area, circ)
        reds   = _detect_tokens(red_m,    min_area, circ)
        yellows= _detect_tokens(yellow_m, min_area, circ)
        for y in yellows:
            y["is_yellow"] = True

        # --- EV2 Police: a blue car on the road -> "collect a red within 5s" --
        police = self._detect_police(hsv, road_region, w, h) if cfg["enable_police_event"] else []
        for c in police:
            c["is_police"] = True
        if police:
            self._police_until = time.time() + cfg["police_grab_secs"]   # (re)arm window
            if cfg.get("save_event_frames") and time.time() - self._last_save > 0.4:
                os.makedirs("police_frames", exist_ok=True)
                cv2.imwrite(f"police_frames/police_{int(time.time()*1000)}.png", front_frame)
                self._last_save = time.time()
        grabbing_red = time.time() < self._police_until

        # --- Plan a path ------------------------------------------------
        if grabbing_red:
            # INVERT: seek the nearest RED (treat it as the reward) while still
            # avoiding the police car + yellows. This is what passes EV2.
            avoid = police + yellows
            if reds:
                target_x = self._field_target(reds, avoid, h, w, center_x)
                mode = "EV2:GRAB-RED"
            else:
                target_x = center_x + self._lane_keep(hsv, roi_mask, w, center_x) * (w * 0.5)
                mode = "EV2:wait-red"
            throttle_hazards = avoid
        else:
            # Normal play. Police shouldn't appear here, but avoid it if it does.
            hazards = reds + yellows + police
            if not greens and not hazards:
                target_x = center_x + self._lane_keep(hsv, roi_mask, w, center_x) * (w * 0.5)
                mode = "LANEKEEP"
            else:
                target_x = self._field_target(greens, hazards, h, w, center_x)
                mode = "PLAN"
            throttle_hazards = hazards

        err = (target_x - center_x) / (w * 0.5)        # -1..1
        steer = self._smooth(_clamp(cfg["steer_gain"] * err))

        # Adaptive throttle: full speed when clear, ease off as the nearest
        # in-path hazard approaches -> gives steering time to dodge.
        accel = self._throttle(throttle_hazards, h, w, center_x)
        if accel < cfg["accel_cruise"] - 1e-3 and not grabbing_red:
            mode = "PLAN!"

        if self._logger is not None:
            self._logger.log(front_frame, steer, accel)

        self._maybe_debug(front_frame, roi_mask,
                          {"green": greens, "red": reds, "yellow": yellows, "police": police},
                          steer, mode, target_x=target_x,
                          masks={"green": green_m, "red": red_m, "yellow": yellow_m})
        return steer, accel

    # -- behaviours ------------------------------------------------------
    def _field_target(self, greens, hazards, h, w, center_x):
        """
        Score candidate lateral target positions and return the best one.
        Greens add reward, hazards subtract a (larger, wider) penalty, both
        weighted by proximity (closer tokens matter more). A mild lane cost
        keeps us from swinging wider than necessary. The argmax is the path
        that grabs the most green while steering clear of red/yellow.
        """
        cfg = self.cfg
        cands = np.linspace(cfg["cand_min_frac"] * w, cfg["cand_max_frac"] * w,
                            cfg["n_candidates"])
        collect_sig = cfg["collect_sigma_frac"] * w
        avoid_sig = cfg["avoid_sigma_frac"] * w

        scores = np.zeros_like(cands)
        for i, cx in enumerate(cands):
            s = 0.0
            for g in greens:
                prox = (g["cy"] / h) ** 2               # near tokens dominate
                s += cfg["green_reward"] * prox * np.exp(-((cx - g["cx"]) ** 2) /
                                                         (2 * collect_sig ** 2))
            for hz in hazards:
                prox = (hz["cy"] / h) ** 2
                if hz.get("is_police"):
                    wgt = cfg["police_weight"]      # collision = GAME OVER -> avoid hard
                elif hz.get("is_yellow"):
                    wgt = cfg["yellow_weight"]
                else:
                    wgt = 1.0
                s -= cfg["hazard_penalty"] * wgt * prox * np.exp(-((cx - hz["cx"]) ** 2) /
                                                                 (2 * avoid_sig ** 2))
            # mild pull toward center so it doesn't wander when lanes are equal
            s -= cfg["lane_cost"] * abs(cx - center_x) / (0.5 * w)
            scores[i] = s
        return float(cands[int(np.argmax(scores))])

    def _detect_police(self, hsv, road_region, w, h):
        """Detect the blue police car on the road (EV2). Cars aren't round, so NO
        circularity filter; require a sizable blob AND that it be in the near half
        of the road, so a distant blue speck can't false-trigger the event."""
        cfg = self.cfg
        m = cv2.bitwise_and(_build_color_mask(hsv, cfg["police"]), road_region)
        blobs = _detect_tokens(m, cfg["police_min_area_frac"] * (w * h), 0.0)
        min_cy = cfg["police_min_cy_frac"] * h
        return [b for b in blobs if b["cy"] >= min_cy]

    def _path_hazard(self, hazards, h, w, center_x):
        """Return the closest hazard that is both near and in our path, else None."""
        near_y = h * self.cfg["hazard_near_y_frac"]
        path_w = w * self.cfg["hazard_path_half_w"]
        in_path = [t for t in hazards
                   if t["cy"] >= near_y and abs(t["cx"] - center_x) <= path_w]
        return _closest(in_path)

    def _throttle(self, hazards, h, w, center_x):
        """Smoothly interpolate cruise->dodge throttle by how close the nearest
        in-path hazard is. No hazard ahead -> full speed."""
        cfg = self.cfg
        threat = self._path_hazard(hazards, h, w, center_x)
        if threat is None:
            return cfg["accel_cruise"]
        near_y = h * cfg["hazard_near_y_frac"]          # where "ahead" starts
        bottom = h * cfg["roi_bottom_y"]                # closest the ROI sees
        # prox: 0 when the hazard just enters the near zone, 1 when it's right on us
        prox = float(np.clip((threat["cy"] - near_y) / max(1.0, bottom - near_y), 0.0, 1.0))
        return cfg["accel_cruise"] + (cfg["accel_dodge"] - cfg["accel_cruise"]) * prox

    def _road_region(self, hsv, roi_mask):
        """Grey-asphalt mask, dilated to cover on-road tokens, ANDed with the ROI.
        Tokens live on this surface; grass (green) does not, so intersecting the
        colour masks with this removes grass-as-green and roadside clutter."""
        cfg = self.cfg
        road = cv2.inRange(
            hsv,
            np.array((0, 0, cfg["road_val_min"]), np.uint8),
            np.array((179, cfg["road_sat_max"], 255), np.uint8),
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (cfg["road_dilate"], cfg["road_dilate"]))
        road = cv2.dilate(road, k)          # grow over tokens sitting on the road
        return cv2.bitwise_and(road, roi_mask)

    def _lane_keep(self, hsv, roi_mask, w, center_x):
        """Gently steer toward the centroid of the grey road inside the ROI."""
        cfg = self.cfg
        road = cv2.inRange(
            hsv,
            np.array((0, 0, cfg["road_val_min"]), np.uint8),
            np.array((179, cfg["road_sat_max"], 255), np.uint8),
        )
        road = cv2.bitwise_and(road, roi_mask)
        M = cv2.moments(road)
        if M["m00"] == 0:
            return 0.0
        road_cx = M["m10"] / M["m00"]
        err = (road_cx - center_x) / (w * 0.5)
        return _clamp(cfg["lanekeep_gain"] * err)

    def _smooth(self, steer):
        a = self.cfg["smoothing"]
        self._prev_steer = a * steer + (1 - a) * self._prev_steer
        return _clamp(self._prev_steer)

    # -- debug overlay ---------------------------------------------------
    def _maybe_debug(self, frame, roi_mask, tokens, steer, mode, target_x=None, masks=None):
        if not self.cfg["debug"]:
            return
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            colors = {"green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255),
                      "police": (255, 0, 0)}
            # Paint what each colour mask actually caught, so a mis-classified
            # token (e.g. a salmon red landing in the green mask) is obvious.
            if masks:
                tint = np.zeros_like(vis)
                for name, m in masks.items():
                    tint[m > 0] = colors[name]
                vis = cv2.addWeighted(vis, 0.65, tint, 0.55, 0)
            cv2.polylines(vis, [_roi_polygon(w, h)], True, (255, 255, 255), 1)
            for name, lst in tokens.items():
                for t in lst:
                    p = (int(t["cx"]), int(t["cy"]))
                    cv2.circle(vis, p, 8, colors[name], 2)
            # chosen path target: a vertical cyan line
            if target_x is not None:
                tx = int(target_x)
                cv2.line(vis, (tx, int(h * 0.45)), (tx, h - 5), (255, 255, 0), 2)
            cx = int(w * self.cfg["center_x_frac"])
            tip = int(cx + steer * w * 0.25)
            cv2.arrowedLine(vis, (cx, h - 5), (tip, h - 40), (255, 0, 255), 3, tipLength=0.3)
            cv2.putText(vis, f"{mode}  steer={steer:+.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.imshow("Agent Debug", vis)
            cv2.waitKey(1)
        except Exception:
            pass

# =========================================================================
# DataLogger -- captures play for future imitation learning (off by default)
# =========================================================================
class DataLogger:
    """Saves (frame, steering, acceleration) so you can train LearnedPolicy."""

    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(self.dir, exist_ok=True)
        self.labels_path = os.path.join(self.dir, "labels.csv")
        if not os.path.exists(self.labels_path):
            with open(self.labels_path, "w") as f:
                f.write("filename,steering,acceleration,timestamp\n")
        self._n = 0

    def log(self, frame, steering, acceleration):
        ts = time.time()
        fname = f"f_{int(ts*1000)}_{self._n:06d}.jpg"
        cv2.imwrite(os.path.join(self.dir, fname), frame)
        with open(self.labels_path, "a") as f:
            f.write(f"{fname},{steering:.4f},{acceleration:.4f},{ts:.3f}\n")
        self._n += 1

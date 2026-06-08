import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes
from collections import deque

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 1.0   # always full throttle; starts at 1.0 so car moves immediately
}
data_lock = threading.Lock()
is_running = True

# ---------------------------------------------------------
# Game State (autonomous logic)
# ---------------------------------------------------------
YELLOW_EFFECT_DURATION = 5.0  # seconds
TOKEN_MIN_AREA    = 200         # px² minimum to count as a token
TOKEN_MAX_AREA    = 200000     # px² maximum — set high; circularity handles grass
TOKEN_CIRCULARITY = 0.30       # 0=any shape, 1=perfect circle; spheres score ~0.7+
ROAD_EDGE_MARGIN  = 0.08       # ignore outermost 8% of frame width on each side
CAR_DETECT_AREA  = 800         # px² edge-contour threshold for back-camera car
LANE_SWITCH_HOLD = 0.7         # seconds to hold hard steer during lane switch
DELAY_SECONDS    = 0.15        # camera / action delay simulation

# Lane-distance thresholds (normalised horizontal error: 0=centre, 1=frame edge)
# Tune these if the road is narrower/wider in the camera FOV.
#   |err| < SAME_LANE_ERR  → token is in the current lane  → steer = 0.0 (already aligned)
#   |err| < FAR_LANE_ERR   → token is ~1 lane away         → proportional steer
#   |err| >= FAR_LANE_ERR  → token is 2+ lanes away        → INSTANT full ±1.0 committed steer
SAME_LANE_ERR    = 0.18   # roughly ½ lane width dead-zone
FAR_LANE_ERR     = 0.42   # lower = commits full steer sooner (was 0.45)
MULTI_LANE_HOLD  = 0.45   # seconds to lock full steer when crossing 2+ lanes to a green token
NUM_LANES        = 5      # number of drivable lanes across the road

game_state = {
    'police_active':      False,
    'fast_car_active':    False,
    'low_brightness':     False,
    # yellow token effects
    'effect_hidden_type':  False,
    'effect_invisible':    False,
    'effect_cam_delay':    False,
    'effect_action_delay': False,
    'effect_corrupted':    False,
    'effect_end':          0.0,
    # lane switch
    'lane_switch_dir': 0.0,
    'lane_switch_end': 0.0,
    # yellow proximity tracker
    'yellow_was_near': False,
    # consecutive-frame counters — event must persist N frames before we act
    'police_frames':   0,
    'fast_car_frames': 0,
    # committed green pursuit — keeps steering toward the last chosen green token
    # for up to 0.5 s even if the token briefly disappears (collected / occluded)
    'target_x':          None,
    'target_commit_end': 0.0,
}
state_lock = threading.Lock()

buf_lock     = threading.Lock()
frame_buffer  = deque(maxlen=120)   # (timestamp, frame)
action_buffer = deque(maxlen=120)   # (timestamp, steering, accel)

_last_print_time  = 0.0   # throttle console prints to once per second
_debug_frame_ref  = {}    # latest raw frame for the HSV click-sampler

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception:
            pass

        while is_running:
            start_time = time.time()
            try:
                self.execute_func()
            except Exception as task_err:
                print(f"[{self.name}] TASK EXCEPTION: {task_err}")
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time

            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
    
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
                
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
                
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
    
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Vision Helpers
# ---------------------------------------------------------

def get_brightness(frame):
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) / 255.0

def on_debug_mouse(event, x, y, _flags, _param):
    """Click any pixel in the Autonomous Debug window to print its HSV value."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    frame = _debug_frame_ref.get('frame')
    if frame is None:
        return
    fh, fw = frame.shape[:2]
    fx = min(int(x * fw / 640), fw - 1)
    fy = min(int(y * fh / 480), fh - 1)
    bgr   = frame[fy, fx]
    hsv_px = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    print(f"[SAMPLER] pos=({fx},{fy})  "
          f"BGR=({int(bgr[0])},{int(bgr[1])},{int(bgr[2])})  "
          f"HSV=H:{hsv_px[0]} S:{hsv_px[1]} V:{hsv_px[2]}  "
          f"<-- paste into detect_tokens HSV range")

def find_blobs(mask, min_area=TOKEN_MIN_AREA, max_area=TOKEN_MAX_AREA):
    """
    Return (cx, cy, area) blobs sorted by y descending (nearest first).
    Grass is filtered out by:
      - max_area cap  (grass blobs are huge)
      - circularity   (tokens are round; grass edges are jagged)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        a = cv2.contourArea(c)
        if not (min_area <= a <= max_area):
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * a / (perimeter ** 2)
        if circularity < TOKEN_CIRCULARITY:
            continue
        M = cv2.moments(c)
        if M['m00'] > 0:
            blobs.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00']), a))
    return sorted(blobs, key=lambda b: -b[1])

def detect_tokens(frame):
    """
    Detect green/red/yellow tokens.
    Tokens are large semi-transparent glossy spheres — mint-green, pinkish-red, gold.
    Use the HSV click-sampler (click in Autonomous Debug window) to tune these ranges.
    """
    h, w = frame.shape[:2]
    roi_top = h // 4          # skip top 25% (sky / scoreboard)
    roi = frame[roi_top:]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    k   = np.ones((5, 5), np.uint8)

    # Road edge mask — strip outer grass bands
    edge      = int(w * ROAD_EDGE_MARGIN)
    road_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    road_mask[:, edge: w - edge] = 255

    # Green — lime/spring/mint/cyan-green spheres.
    # H=40-90 covers yellow-green through pure-green to cyan-green in OpenCV scale.
    # S and V lowered so semi-transparent glowing spheres are not missed.
    # Circularity filter removes grass which has the same hue but jagged edges.
    gm = cv2.morphologyEx(
        cv2.inRange(hsv, np.array([40, 40, 80]), np.array([90, 255, 255])),
        cv2.MORPH_OPEN, k)
    gm = cv2.bitwise_and(gm, road_mask)

    # Red — pinkish-red semi-transparent spheres  H=0-15 + H=160-180
    # Low S minimum to catch pale/pastel reds; V>=80 keeps dark background out.
    rm = cv2.morphologyEx(
        cv2.inRange(hsv, np.array([0,   20, 80]), np.array([15,  160, 255])) |
        cv2.inRange(hsv, np.array([160, 20, 80]), np.array([180, 160, 255])),
        cv2.MORPH_OPEN, k)
    rm = cv2.bitwise_and(rm, road_mask)

    # Yellow/gold — H=15-38 keeps it below green range; medium S, V>=80
    ym = cv2.morphologyEx(
        cv2.inRange(hsv, np.array([15, 80, 80]), np.array([38, 255, 255])),
        cv2.MORPH_OPEN, k)
    ym = cv2.bitwise_and(ym, road_mask)

    def adj(blobs):
        return [(cx, cy + roi_top, a) for cx, cy, a in blobs]

    return {'green': adj(find_blobs(gm)), 'red': adj(find_blobs(rm)),
            'yellow': adj(find_blobs(ym)), 'w': w, 'h': h}

def detect_back_events(back_frame):
    """
    Returns (police_raw, fast_car_raw) — raw per-frame signals.
    Callers use frame counters to confirm before treating as real events.
    """
    if back_frame is None:
        return False, False
    h, w = back_frame.shape[:2]

    # Only the lower 40% of the back frame — a following car is close and low.
    # Sky / buildings are in the top portion and would cause false positives.
    lower = back_frame[int(h * 0.60):]
    hsv   = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)

    # Police lights: extremely vivid, bright blue (S>180, V>180).
    # Night-sky blue and neon signs are far less saturated/bright → won't match.
    blue_mask = cv2.inRange(hsv, np.array([105, 180, 180]), np.array([135, 255, 255]))
    police_raw = cv2.countNonZero(blue_mask) > 600

    # Fast car: compact blob with large area in the horizontal centre.
    # Road markings are thin lines (small contour area); a car is a solid block.
    gray    = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    roi     = gray[:, w // 4: 3 * w // 4]
    edges   = cv2.Canny(roi, 60, 160)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max((cv2.contourArea(c) for c in cnts), default=0)
    fast_car_raw = (not police_raw) and (max_area > CAR_DETECT_AREA * 3)

    return police_raw, fast_car_raw

def steer_toward(target_x, frame_w, target_y=None, frame_h=None):
    """
    Three-zone lane-aware steering toward a target.
      Zone 1 – same lane   (|err| < SAME_LANE_ERR):  steer = 0.0  (already aligned)
      Zone 2 – ~1 lane     (SAME_LANE_ERR .. FAR_LANE_ERR): proportional
      Zone 3 – 2+ lanes    (|err| >= FAR_LANE_ERR):  steer = ±1.0 (full immediate steer)
    """
    err = (target_x - frame_w / 2) / (frame_w / 2)
    if abs(err) < SAME_LANE_ERR:
        return 0.0                                    # already in the target lane
    if abs(err) >= FAR_LANE_ERR:
        return float(np.sign(err))                    # 2+ lanes away → full steer
    # ~1 lane away: proportional with proximity ramp
    gain = 4.0
    if target_y is not None and frame_h is not None:
        gain += (target_y / frame_h) * 2.0
    return float(np.clip(err * gain, -1.0, 1.0))

def steer_away(target_x, frame_w, target_y=None, frame_h=None):
    """
    Full opposite steer when the token is within 50% of centre (directly in our path).
    Proportional when we are already mostly clear of it.
    """
    err = (target_x - frame_w / 2) / (frame_w / 2)
    if abs(err) < 0.50:                              # token is in / near our lane → full dodge
        return -1.0 if err >= 0 else 1.0
    gain = 3.5
    if target_y is not None and frame_h is not None:
        gain += (target_y / frame_h) * 2.0
    return float(np.clip(-err * gain, -1.0, 1.0))

def find_best_green(green_tokens, red_tokens, yellow_tokens, frame_w):
    """
    Return the nearest green token in a lane that contains NO red or yellow token.
    An entire lane is considered dangerous if any red/yellow sits in it,
    regardless of how close that token is.
    Falls back to the nearest green if every lane is occupied by a bad token.
    """
    if not green_tokens:
        return None
    danger_lanes = set(get_lane(t[0], frame_w) for t in red_tokens + yellow_tokens)
    for g in green_tokens:               # already sorted nearest-first (highest y first)
        if get_lane(g[0], frame_w) not in danger_lanes:
            return g
    return green_tokens[0]               # every lane is dangerous — pick nearest anyway

def get_lane(x, frame_w):
    """Return 0-indexed lane number (0 = leftmost) for pixel x in a frame of width frame_w."""
    edge   = int(frame_w * ROAD_EDGE_MARGIN)
    road_w = frame_w - 2 * edge
    lane_w = road_w / NUM_LANES
    lane   = int((x - edge) / lane_w)
    return max(0, min(NUM_LANES - 1, lane))

def draw_debug(frame, tokens, steer, gs):
    _debug_frame_ref['frame'] = frame
    dbg = frame.copy()
    h, w = dbg.shape[:2]

    edge   = int(w * ROAD_EDGE_MARGIN)
    road_w = w - 2 * edge
    lane_w = road_w / NUM_LANES

    # --- Road edges (solid grey) ---
    cv2.line(dbg, (edge, 0),     (edge, h),     (110, 110, 110), 1)
    cv2.line(dbg, (w - edge, 0), (w - edge, h), (110, 110, 110), 1)

    # --- Lane dividers (dashed white) ---
    for i in range(1, NUM_LANES):
        lx = int(edge + i * lane_w)
        y  = 0
        while y < h:
            cv2.line(dbg, (lx, y), (lx, min(y + 16, h)), (190, 190, 190), 1)
            y += 32

    # --- Lane number labels + live score display ---
    lane_scores = [0.0] * NUM_LANES
    for cx, cy, _ in tokens['green']:
        lane_scores[get_lane(cx, w)] += 10.0 * (1.0 + cy / h)
    for cx, cy, _ in tokens['red']:
        lane_scores[get_lane(cx, w)] -= 20.0 * (1.0 + cy / h)
    for cx, cy, _ in tokens['yellow']:
        lane_scores[get_lane(cx, w)] -= 10.0 * (1.0 + cy / h)

    for i in range(NUM_LANES):
        lx_c = int(edge + (i + 0.5) * lane_w)
        cv2.putText(dbg, f'L{i}', (lx_c - 9, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        sc    = lane_scores[i]
        scol  = (60, 210, 60) if sc > 0 else (60, 60, 210) if sc < 0 else (130, 130, 130)
        cv2.putText(dbg, f'{sc:+.0f}', (lx_c - 14, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, scol, 1)

    # --- Faint red overlay on every lane that contains a red token ---
    red_lanes = set(get_lane(t[0], w) for t in tokens['red'])
    for rl in red_lanes:
        rx0  = int(edge + rl * lane_w) + 1
        rx1  = int(edge + (rl + 1) * lane_w) - 1
        over = dbg.copy()
        cv2.rectangle(over, (rx0, 0), (rx1, h), (0, 0, 200), -1)
        cv2.addWeighted(over, 0.20, dbg, 0.80, 0, dbg)

    # --- Faint green highlight for the current target lane ---
    tgt_x   = gs.get('target_x')
    tgt_end = gs.get('target_commit_end', 0.0)
    if tgt_x is not None and time.time() < tgt_end:
        tl   = get_lane(tgt_x, w)
        tx0  = int(edge + tl * lane_w) + 1
        tx1  = int(edge + (tl + 1) * lane_w) - 1
        over = dbg.copy()
        cv2.rectangle(over, (tx0, 0), (tx1, h), (0, 140, 0), -1)
        cv2.addWeighted(over, 0.18, dbg, 0.82, 0, dbg)

    # --- Car position arrow at the bottom (car is always at frame centre) ---
    car_lane = get_lane(w // 2, w)
    cv2.arrowedLine(dbg, (w // 2, h - 2), (w // 2, h - 30),
                    (0, 220, 255), 2, tipLength=0.45)
    cv2.putText(dbg, f'CAR L{car_lane}', (w // 2 - 24, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

    # --- Token markers — circle + colour-initial + lane index ---
    for cx, cy, _ in tokens['green']:
        ln = get_lane(cx, w)
        cv2.circle(dbg, (cx, cy), 14, (0, 255, 0), 2)
        cv2.putText(dbg, f'G{ln}', (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
    for cx, cy, _ in tokens['red']:
        ln = get_lane(cx, w)
        cv2.circle(dbg, (cx, cy), 14, (0, 0, 255), 2)
        cv2.putText(dbg, f'R{ln}', (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
    for cx, cy, _ in tokens['yellow']:
        ln = get_lane(cx, w)
        cv2.circle(dbg, (cx, cy), 14, (0, 200, 255), 2)
        cv2.putText(dbg, f'Y{ln}', (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)

    # --- Status bar ---
    status_flags = [k.replace('effect_', '').upper() for k, v in gs.items()
                    if k.startswith('effect_') and v is True and k != 'effect_end']
    if gs['police_active']:   status_flags.insert(0, 'POLICE')
    if gs['fast_car_active']: status_flags.insert(0, 'FAST-CAR')
    if gs['low_brightness']:  status_flags.insert(0, 'DARK')

    cv2.putText(dbg, f"Steer: {steer:+.2f}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
    cv2.putText(dbg, ' | '.join(status_flags) if status_flags else 'OK',
                (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.putText(dbg, 'Click pixel = print HSV',
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

    cv2.imshow("Autonomous Debug", cv2.resize(dbg, (640, 480)))
    cv2.setMouseCallback("Autonomous Debug", on_debug_mouse)
    cv2.waitKey(1)

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key):
    #This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return
        
    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return
            
        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes
            
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
                
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet
                
            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes
                
        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame
                
                # You may disable this if you don't need to display the frames / This could effect the fps
                frame_resized = cv2.resize(frame, (640, 480))
                cv2.imshow(window_name, frame_resized)
                cv2.waitKey(1)
                
    except Exception:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

def processing_task():
    with data_lock:
        raw_front  = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']

    if raw_front is None:
        return

    now = time.time()

    # Always buffer raw frames so the cam-delay effect can replay them
    with buf_lock:
        frame_buffer.append((now, raw_front))

    # Read active effects (short lock just for reads)
    with state_lock:
        eff_end       = game_state['effect_end']
        eff_cam_delay = game_state['effect_cam_delay']
        eff_corrupted = game_state['effect_corrupted']

    # Select working frame ------------------------------------------------------
    if eff_cam_delay and now < eff_end:
        target_ts = now - DELAY_SECONDS
        with buf_lock:
            frames = list(frame_buffer)
        front_frame = next(
            (f for ts, f in reversed(frames) if ts <= target_ts), raw_front)
    else:
        front_frame = raw_front

    if eff_corrupted and now < eff_end:
        front_frame = cv2.GaussianBlur(front_frame, (7, 7), 0)

    # Perception ----------------------------------------------------------------
    brightness       = get_brightness(front_frame)
    tokens           = detect_tokens(front_frame)
    police, fast_car = detect_back_events(back_frame)

    w, h = tokens['w'], tokens['h']

    # Decision ------------------------------------------------------------------
    with state_lock:
        gs   = game_state
        now2 = time.time()

        # 0.15 threshold: normal night sky ~0.37 stays False;
        # only the explicit game blackout event (much darker) triggers True
        # 0.15 threshold: normal night sky ~0.37 stays False;
        # only the explicit game blackout event (much darker) triggers True
        gs['low_brightness'] = brightness < 0.15

        # Police — require 8 consecutive frames of raw detection before confirming.
        # This filters single-frame glitches from neon lights / reflections.
        if police:
            gs['police_frames'] += 1
        else:
            # Decay faster (2 per miss) so false blips clear quickly
            gs['police_frames'] = max(0, gs['police_frames'] - 2)
        gs['police_active'] = gs['police_frames'] >= 14

        # Fast car — require 6 consecutive frames, then trigger one-shot lane switch.
        if fast_car:
            gs['fast_car_frames'] += 1
        else:
            gs['fast_car_frames'] = max(0, gs['fast_car_frames'] - 1)

        fast_car_confirmed = gs['fast_car_frames'] >= 6
        if fast_car_confirmed and not gs['fast_car_active']:
            gs['fast_car_active']  = True
            gs['lane_switch_dir']  = -1.0 if gs['lane_switch_dir'] >= 0 else 1.0
            gs['lane_switch_end']  = now2 + LANE_SWITCH_HOLD
            print("[Event] Fast car confirmed — switching lanes")
        elif not fast_car_confirmed:
            gs['fast_car_active'] = False

        # Expire yellow effects
        if now2 >= gs['effect_end']:
            for k in ('effect_hidden_type', 'effect_invisible', 'effect_cam_delay',
                      'effect_action_delay', 'effect_corrupted'):
                gs[k] = False

        # Detect yellow token collection (blob disappears near car centre)
        bottom_y  = h * 0.80
        cx_margin = w * 0.25
        yellow_near_now = any(
            cy > bottom_y and abs(cx - w // 2) < cx_margin
            for cx, cy, _ in tokens['yellow']
        )
        if not yellow_near_now and gs['yellow_was_near']:
            gs['yellow_was_near'] = False
            if now2 >= gs['effect_end']:          # don't stack effects
                rng = int(now2 * 1000) % 5
                gs['effect_end'] = now2 + YELLOW_EFFECT_DURATION
                effect_keys = ['effect_hidden_type', 'effect_invisible',
                               'effect_cam_delay', 'effect_action_delay', 'effect_corrupted']
                gs[effect_keys[rng]] = True
                print(f"[Yellow] {effect_keys[rng]} active for {YELLOW_EFFECT_DURATION}s")
        else:
            gs['yellow_was_near'] = yellow_near_now

        # Reclassify tokens based on active conditions BEFORE making any decision
        # Rule: hidden_type effect → can't see colour → treat all as dangerous
        if gs['effect_hidden_type'] and now2 < gs['effect_end']:
            tokens['yellow'] = tokens['green'] + tokens['red'] + tokens['yellow']
            tokens['green']  = []
            tokens['red']    = []

        # Rule: darkness → ALL tokens become yellow debuffs regardless of colour.
        # This means police can't find a red token during darkness — it waits for light.
        if gs['low_brightness']:
            tokens['yellow'] = tokens['green'] + tokens['red'] + tokens['yellow']
            tokens['green']  = []
            tokens['red']    = []

        # Steering priority order -----------------------------------------------
        if now2 < gs['lane_switch_end']:
            steer = gs['lane_switch_dir']                     # hold lane-switch steer

        elif gs['effect_invisible'] and now2 < gs['effect_end']:
            steer = 0.0                                       # tokens invisible — drive straight

        elif gs['police_active'] and tokens['red']:
            # Police active → must take nearest red token
            t = max(tokens['red'], key=lambda b: b[1])
            steer = steer_toward(t[0], w, t[1], h)

        else:
            # --- Lane-scoring decision ---
            # Every token in every lane contributes to that lane's score.
            # Proximity weight: tokens further down the frame (closer to the car)
            # have more influence than distant ones.
            #   Green  → +10 × (1 + proximity)   attract
            #   Red    → -20 × (1 + proximity)   repel hard
            #   Yellow → -10 × (1 + proximity)   repel soft
            edge_px = int(w * ROAD_EDGE_MARGIN)
            lane_px = (w - 2 * edge_px) / NUM_LANES
            scores  = [0.0] * NUM_LANES

            for cx, cy, _ in tokens['green']:
                scores[get_lane(cx, w)] += 10.0 * (1.0 + cy / h)
            for cx, cy, _ in tokens['red']:
                scores[get_lane(cx, w)] -= 20.0 * (1.0 + cy / h)
            for cx, cy, _ in tokens['yellow']:
                scores[get_lane(cx, w)] -= 10.0 * (1.0 + cy / h)

            best_lane  = max(range(NUM_LANES), key=lambda l: scores[l])
            best_score = scores[best_lane]
            best_cx    = int(edge_px + (best_lane + 0.5) * lane_px)

            if all(s == 0.0 for s in scores):
                # No tokens anywhere — use committed target or drift to centre
                if gs['target_x'] is not None and now2 < gs['target_commit_end']:
                    steer = steer_toward(gs['target_x'], w)
                else:
                    last_steer = shared_data.get('steering_input', 0.0)
                    steer = float(np.clip(-last_steer * 0.3, -0.3, 0.3))

            elif best_score < 0.0:
                # Every lane has at least one bad token — flee the worst lane
                worst_lane = min(range(NUM_LANES), key=lambda l: scores[l])
                worst_cx   = int(edge_px + (worst_lane + 0.5) * lane_px)
                steer = steer_away(worst_cx, w)

            else:
                # Steer toward the highest-scoring lane
                if best_score > 0.0:
                    gs['target_x']          = best_cx
                    gs['target_commit_end'] = now2 + 0.5
                err = (best_cx - w / 2) / (w / 2)
                if abs(err) < SAME_LANE_ERR:
                    steer = 0.0                              # already in best lane
                elif abs(err) >= FAR_LANE_ERR:
                    gs['lane_switch_dir'] = float(np.sign(err))
                    gs['lane_switch_end'] = now2 + MULTI_LANE_HOLD
                    steer = gs['lane_switch_dir']
                else:
                    steer = steer_toward(best_cx, w)

    # Update shared_data OUTSIDE state_lock to avoid lock-order deadlock
    # (send_controls_task acquires data_lock then state_lock — opposite order)
    with data_lock:
        shared_data['steering_input']     = steer
        shared_data['acceleration_input'] = 1.0

    # Buffer action for action-delay effect
    with buf_lock:
        action_buffer.append((now2, steer, 1.0))

    draw_debug(front_frame, tokens, steer, gs)

    # --- Periodic diagnostics (once per second) ---
    global _last_print_time
    if now2 - _last_print_time >= 1.0:
        _last_print_time = now2
        print("=" * 55)
        print(f"[CAR]  steering={steer:+.3f}  acceleration=1.000")
        print(f"[CAM]  front_frame={'OK' if raw_front is not None else 'NONE'}"
              f"  back_frame={'OK' if back_frame is not None else 'NONE'}")
        print(f"[SENS] brightness={brightness:.3f}  low_brightness={gs['low_brightness']}")
        print(f"[TOKS] green={len(tokens['green'])}  red={len(tokens['red'])}  yellow={len(tokens['yellow'])}")
        print(f"[EVT]  police={gs['police_active']}  fast_car={gs['fast_car_active']}")
        print(f"[FX]   hidden_type={gs['effect_hidden_type']}  invisible={gs['effect_invisible']}"
              f"  cam_delay={gs['effect_cam_delay']}")
        print(f"       action_delay={gs['effect_action_delay']}  corrupted={gs['effect_corrupted']}"
              f"  ends_in={max(0.0, gs['effect_end'] - now2):.1f}s")
        print(f"[LANE] switch_active={now2 < gs['lane_switch_end']}  dir={gs['lane_switch_dir']:+.1f}")
        print(f"[NET]  control_conn={'connected' if control_conn is not None else 'NONE'}"
              f"  front_sock={'connected' if front_camera_sock is not None else 'NONE'}"
              f"  back_sock={'connected' if back_camera_sock is not None else 'NONE'}")

def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    now = time.time()

    with data_lock:
        steering_input     = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']

    # When action-delay effect is active, replay the action from ~150 ms ago
    with state_lock:
        eff_action_delay = game_state['effect_action_delay']
        eff_end          = game_state['effect_end']

    if eff_action_delay and now < eff_end:
        target_ts = now - DELAY_SECONDS
        with buf_lock:
            buf = list(action_buffer)
        for ts, s, a in reversed(buf):
            if ts <= target_ts:
                steering_input, acceleration_input = s, a
                break

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    
    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    # This is where you define tasks with explicit Scheduling parameters (Concurrency, Priority, Period)
    # Period refers to the period of execution of the task in seconds
    # Priority refers to the priority of the task, higher priority means higher priority
    # Concurrency refers to the number of instances of the task that can run at the same time
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()
    
    try:
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()
    
    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")

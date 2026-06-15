import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

BRIGHTNESS_THRESHOLD = 80.0

# ---------------------------------------------------------
# YOLO Configuration (Phase 5)
# Set YOLO_ENABLED = True after training is complete.
# ---------------------------------------------------------
YOLO_ENABLED    = False
YOLO_MODEL_PATH = 'runs/detect/rtse_detector/weights/best.pt'
YOLO_CONF       = 0.5
yolo_model      = None

TOKEN_HSV = {
    'green':  ([40, 80,  80],  [80,  255, 255]),
    'yellow': ([20, 100, 100], [35,  255, 255]),
    'red':    ([0,   80, 150], [15,  195, 255]),  # S cap at 195 excludes curb (S=255); coin is S=133
}

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame':  None,
    'steering_input':     0.0,
    'acceleration_input': 0.0,
    'debug_frame':        None,
    'debug_back_frame':   None,
    'drive_mode':         'NORMAL',
    'brightness':         0.0,
    'rear_event':         'clear',
}
data_lock = threading.Lock()
is_running = True

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
            except Exception as e:
                print(f"[{self.name}] EXCEPTION (thread kept alive): {e}")
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

def compute_brightness(frame):
    return float(np.mean(frame))


_lane_ema = {'left': None, 'right': None}  # temporal smoothing of lane positions

def detect_lane_centers(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.35):int(h * 0.82), :]
    # Blur before HSV conversion to reduce noise speckles
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv_roi = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_roi,
                             np.array([0,   0, 190]),
                             np.array([180, 45, 255]))
    # Morphological opening removes isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    col_sums = np.sum(white_mask, axis=0).astype(np.float32)
    # Smooth column sums so the dominant lane band wins over isolated spikes
    col_sums = np.convolve(col_sums, np.ones(15) / 15, mode='same')
    mid = w // 2
    left_half  = col_sums[:mid]
    right_half = col_sums[mid:]
    raw_left  = int(np.argmax(left_half))       if np.max(left_half)  > 50 else None
    raw_right = int(np.argmax(right_half)) + mid if np.max(right_half) > 50 else None
    # EMA smoothing across frames; fall back to frame edges when no lane found
    alpha = 0.4
    if raw_left is not None:
        _lane_ema['left'] = raw_left if _lane_ema['left'] is None else int(alpha * raw_left + (1 - alpha) * _lane_ema['left'])
    if raw_right is not None:
        _lane_ema['right'] = raw_right if _lane_ema['right'] is None else int(alpha * raw_right + (1 - alpha) * _lane_ema['right'])
    left_x  = _lane_ema['left']  if _lane_ema['left']  is not None else 0
    right_x = _lane_ema['right'] if _lane_ema['right'] is not None else w
    mid_x   = (left_x + right_x) // 2
    return left_x, right_x, mid_x


def compute_steering(target_x, frame_width):
    offset = (target_x - frame_width / 2) / (frame_width / 2)
    return float(np.clip(offset, -1.0, 1.0))


def detect_tokens(frame):
    h, w = frame.shape[:2]
    # Build exclusion mask so we don't detect the player's own car or game UI
    exclude = np.zeros((h, w), dtype=np.uint8)
    exclude[int(h * 0.80):,  int(w * 0.30):int(w * 0.70)] = 255  # player car (bottom-center)
    exclude[:int(h * 0.22),  int(w * 0.65):]               = 255  # score/distance HUD (top-right)
    exclude[:,                :int(w * 0.38)]               = 255  # left curb / road markers
    exclude[:,                int(w * 0.85):]               = 255  # right curb / road edge

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    results = {}
    for color, (lo, hi) in TOKEN_HSV.items():
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        if color == 'red':
            mask |= cv2.inRange(hsv, np.array([170, 80, 150]),
                                     np.array([180, 195, 255]))
        mask[exclude > 0] = 0
        # Morphological opening: removes noise specks, keeps solid blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 80:  # raised threshold: ignore tiny noise
                continue
            # Circularity filter — tokens are coin-shaped; elongated road markings are not
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0 and (4 * np.pi * area / perimeter ** 2) < 0.25:
                continue
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                blobs.append((cx, cy, int(np.sqrt(area / np.pi))))
        results[color] = blobs
    return results


_rear_history = ['clear'] * 5  # rolling vote buffer — smooths single-frame false detections
_rear_idx = [0]

def detect_rear_event(back_frame):
    if back_frame is None:
        raw = 'clear'
    else:
        bh, bw = back_frame.shape[:2]
        roi = back_frame[int(bh * 0.30):int(bh * 0.85), int(bw * 0.15):int(bw * 0.85)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([100, 150, 150]), np.array([130, 255, 255]))
        red1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([10,  255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
        red  = cv2.bitwise_or(red1, red2)
        b_contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        r_contours, _ = cv2.findContours(red,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_blue_area = max((cv2.contourArea(c) for c in b_contours), default=0)
        max_red_area  = max((cv2.contourArea(c) for c in r_contours), default=0)
        # Relaxed from 200→150 each: police lights alternate, so one colour sometimes dominates
        if max_blue_area > 150 and max_red_area > 150:
            raw = 'police'
        else:
            yellow_mask = cv2.inRange(hsv, np.array([15,  80,  80]), np.array([40,  255, 255]))
            green_mask  = cv2.inRange(hsv, np.array([40,  80,  80]), np.array([80,  255, 255]))
            red_t1      = cv2.inRange(hsv, np.array([0,   80,  80]), np.array([10,  255, 255]))
            red_t2      = cv2.inRange(hsv, np.array([170, 80,  80]), np.array([180, 255, 255]))
            token_mask  = yellow_mask | green_mask | red_t1 | red_t2
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Otsu picks the threshold automatically — more robust than fixed 100 under varying light
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh[token_mask > 0] = 0
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw = 'car' if any(cv2.contourArea(c) > 800 for c in contours) else 'clear'

    # Majority vote over last 5 frames — eliminates single-frame noise
    _rear_history[_rear_idx[0] % len(_rear_history)] = raw
    _rear_idx[0] += 1
    return max(set(_rear_history), key=_rear_history.count)


_mode_state = {'mode': 'NORMAL', 'count': 0}
_MODE_CONFIRM = 3  # frames a candidate mode must persist before we commit to it

def decide_action(brightness, tokens, rear_event, mid_x, frame_w, frame_h):
    # --- Mode with hysteresis: ignore single-frame flickers ---
    candidate = 'NORMAL'
    if   rear_event == 'police':            candidate = 'POLICE'
    elif rear_event == 'car':               candidate = 'FAST_CAR'
    elif brightness < BRIGHTNESS_THRESHOLD: candidate = 'DARK'

    if candidate == _mode_state['mode']:
        _mode_state['count'] = min(_mode_state['count'] + 1, _MODE_CONFIRM)
    else:
        _mode_state['count'] -= 1
        if _mode_state['count'] <= 0:
            _mode_state['mode']  = candidate
            _mode_state['count'] = 1
    mode = _mode_state['mode']

    steering = compute_steering(mid_x, frame_w)
    accel    = 1.0

    greens  = tokens.get('green',  [])
    dangers = tokens.get('yellow', []) + tokens.get('red', [])
    # Only react to dangers in the bottom 45 % of the frame (close range)
    ahead   = [t for t in dangers if t[1] > frame_h * 0.55]

    def _dodge(token):
        """Proportional dodge: scale push strength by how close the token is."""
        side      = 1.0 if token[0] < frame_w // 2 else -1.0   # +1 = dodge right, -1 = dodge left
        proximity = (token[1] - frame_h * 0.55) / (frame_h * 0.45 + 1e-6)  # 0..1
        return float(np.clip(side * (0.5 + 0.5 * proximity), -1.0, 1.0))

    if mode == 'POLICE':
        if greens:
            target   = max(greens, key=lambda t: t[1])
            steering = compute_steering(target[0], frame_w)
        if ahead:
            nearest_d       = max(ahead, key=lambda t: t[1])
            nearest_green_y = max((g[1] for g in greens), default=0)
            if nearest_d[1] >= nearest_green_y:
                steering = _dodge(nearest_d)
        accel = 1.0

    elif mode == 'FAST_CAR':
        # Move to the opposite side of the lane centre proportionally
        steering = float(np.clip(-compute_steering(mid_x, frame_w) * 1.5, -1.0, 1.0))
        accel    = 1.0

    elif mode == 'DARK':
        accel = 0.5

    else:  # NORMAL
        if greens:
            target   = max(greens, key=lambda t: t[1])
            steering = compute_steering(target[0], frame_w)
        if ahead:
            nearest_d       = max(ahead, key=lambda t: t[1])
            nearest_green_y = max((g[1] for g in greens), default=0)
            if nearest_d[1] >= nearest_green_y:
                steering = _dodge(nearest_d)
        accel = 0.85 if ahead else 1.0

    return steering, accel, mode

# ---------------------------------------------------------
# YOLO Helpers (Phase 5)
# ---------------------------------------------------------

def load_yolo():
    global yolo_model
    if not YOLO_ENABLED:
        return
    try:
        from ultralytics import YOLO as _YOLO
        yolo_model = _YOLO(YOLO_MODEL_PATH)
        print(f"[YOLO] Model loaded: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[YOLO] Failed to load model: {e}")


def detect_tokens_yolo(frame):
    results = yolo_model(frame, verbose=False, conf=YOLO_CONF)[0]
    tokens = {'green': [], 'red': [], 'yellow': []}
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        r  = int(max(x2 - x1, y2 - y1) / 2)
        if   cls == 0: tokens['green'].append((cx, cy, r))
        elif cls == 1: tokens['red'].append((cx, cy, r))
        elif cls == 2: tokens['yellow'].append((cx, cy, r))
    return tokens


def detect_rear_event_yolo(back_frame):
    if back_frame is None:
        return 'clear'
    results = yolo_model(back_frame, verbose=False, conf=YOLO_CONF)[0]
    labels  = [int(b.cls[0]) for b in results.boxes]
    if 3 in labels: return 'police'
    if 4 in labels: return 'car'
    return 'clear'

# ---------------------------------------------------------
# Steering Smoothing (EMA — prevents rapid ±1 oscillation)
# ---------------------------------------------------------
_steer_ema = [0.0]

def smooth_steering(raw, alpha=0.35):
    """Exponential moving average. alpha=0.35 balances responsiveness and stability."""
    _steer_ema[0] = alpha * raw + (1.0 - alpha) * _steer_ema[0]
    return float(_steer_ema[0])

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

def read_single_camera(sock, data_key):
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

    except Exception:
        pass


def read_front_camera_task():
    read_single_camera(front_camera_sock, 'latest_front_frame')


def read_back_camera_task():
    read_single_camera(back_camera_sock, 'latest_back_frame')


def processing_task():
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame  = shared_data['latest_back_frame']

    if front_frame is None:
        return

    h, w = front_frame.shape[:2]

    brightness              = compute_brightness(front_frame)
    left_x, right_x, mid_x = detect_lane_centers(front_frame)
    if YOLO_ENABLED and yolo_model is not None:
        tokens     = detect_tokens_yolo(front_frame)
        rear_event = detect_rear_event_yolo(back_frame)
    else:
        tokens     = detect_tokens(front_frame)
        rear_event = detect_rear_event(back_frame)
    steering, accel, mode   = decide_action(brightness, tokens, rear_event, mid_x, w, h)
    steering = smooth_steering(steering)

    with data_lock:
        shared_data['steering_input']     = steering
        shared_data['acceleration_input'] = accel
        shared_data['drive_mode']         = mode
        shared_data['brightness']         = brightness
        shared_data['rear_event']         = rear_event

    # Build debug overlay
    debug = front_frame.copy()

    # Lane center lines
    cv2.line(debug, (left_x,  0), (left_x,  h), (0, 255, 0),   2)
    cv2.line(debug, (right_x, 0), (right_x, h), (0, 255, 0),   2)
    cv2.line(debug, (mid_x,   0), (mid_x,   h), (255, 255, 0), 2)

    # Token circles
    TOKEN_DRAW_COLORS = {'green': (0, 255, 0), 'yellow': (0, 165, 255), 'red': (0, 0, 255)}
    for color, blobs in tokens.items():
        bgr = TOKEN_DRAW_COLORS[color]
        for cx, cy, r in blobs:
            cv2.circle(debug, (cx, cy), max(r, 8), bgr, 2)

    # HUD
    steer_dir = 'LEFT' if steering < -0.1 else ('RIGHT' if steering > 0.1 else 'STRAIGHT')
    dark_tag  = '  [DARK]' if mode == 'DARK' else ''
    hud = [
        (f"MODE: {mode}",                                                         (0, 255, 255)),
        (f"Steer: {steering:+.2f}  ({steer_dir})",                               (200, 200, 200)),
        (f"Brightness: {brightness:.1f}{dark_tag}",                               (200, 200, 200)),
        (f"Tokens  G:{len(tokens.get('green',[]))}  "
         f"Y:{len(tokens.get('yellow',[]))}  R:{len(tokens.get('red',[]))}",     (200, 200, 200)),
        (f"Rear: {rear_event}",                                                   (200, 200, 200)),
    ]
    for i, (text, color) in enumerate(hud):
        cv2.putText(debug, text, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    with data_lock:
        shared_data['debug_frame'] = debug

    # Build back camera debug overlay
    if back_frame is not None:
        bh, bw    = back_frame.shape[:2]
        back_debug = back_frame.copy()
        # Show the ROI rectangle where police/car detection runs
        cv2.rectangle(back_debug,
                      (int(bw * 0.15), int(bh * 0.30)),
                      (int(bw * 0.85), int(bh * 0.85)),
                      (0, 255, 255), 1)
        label_color = {'police': (0, 0, 255), 'car': (0, 165, 255), 'clear': (0, 255, 0)}
        cv2.putText(back_debug, f"Rear: {rear_event.upper()}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    label_color.get(rear_event, (200, 200, 200)), 1)
        cv2.putText(back_debug, f"Mode: {mode}",
                    (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        with data_lock:
            shared_data['debug_back_frame'] = back_debug


_ctrl_diag = {'last_print': 0.0, 'sends': 0, 'nonzero_steer': 0}

def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    with data_lock:
        steering_input     = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
        _ctrl_diag['sends'] += 1
        if abs(steering_input) > 0.05:
            _ctrl_diag['nonzero_steer'] += 1
        now = time.time()
        if now - _ctrl_diag['last_print'] >= 1.0:
            print(f"[Controls] steer={steering_input:+.3f}  accel={acceleration_input:+.3f}"
                  f"  | sends/s={_ctrl_diag['sends']}  nonzero_steer/s={_ctrl_diag['nonzero_steer']}")
            _ctrl_diag['sends'] = 0
            _ctrl_diag['nonzero_steer'] = 0
            _ctrl_diag['last_print'] = now
    except Exception as e:
        print(f"[Controls] SEND ERROR — connection lost: {e}")
        control_conn = None


_hsv_raw   = [None]   # latest raw front frame for click inspector
_win_ready = [False]  # whether the mouse callback has been registered

def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and _hsv_raw[0] is not None:
        frame = _hsv_raw[0]
        fh, fw = frame.shape[:2]
        ox = int(x * fw / 640)
        oy = int(y * fh / 480)
        if 0 <= ox < fw and 0 <= oy < fh:
            bgr = frame[oy, ox]
            hsv = cv2.cvtColor(frame[oy:oy+1, ox:ox+1], cv2.COLOR_BGR2HSV)[0, 0]
            print(f"[HSV] click ({ox},{oy})  BGR={tuple(int(v) for v in bgr)}"
                  f"  HSV=({int(hsv[0])}, {int(hsv[1])}, {int(hsv[2])})")

def display_task():
    with data_lock:
        frame      = shared_data['debug_frame']
        back_debug = shared_data['debug_back_frame']
        raw        = shared_data['latest_front_frame']
    if frame is not None:
        if not _win_ready[0]:
            cv2.namedWindow("Debug Frame")
            cv2.setMouseCallback("Debug Frame", _on_mouse)
            _win_ready[0] = True
        if raw is not None:
            _hsv_raw[0] = raw
        cv2.imshow("Debug Frame", cv2.resize(frame, (640, 480)))
        cv2.waitKey(1)
    if back_debug is not None:
        cv2.imshow("Back Camera", cv2.resize(back_debug, (320, 240)))
        cv2.waitKey(1)


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    load_yolo()
    print("Initializing RTSE Sample Drive...")

    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")

    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH,   execute_func=read_front_camera_task)
    t_back_camera  = RTTask("ReadBackCamera",  period=0.005, priority=TaskPriority.HIGH,   execute_func=read_back_camera_task)
    t_processing   = RTTask("Processing",      period=0.030, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls     = RTTask("SendControls",    period=0.010, priority=TaskPriority.HIGH,   execute_func=send_controls_task)

    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()

    try:
        while is_running:
            display_task()   # cv2.imshow must run on the main thread
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()

    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")

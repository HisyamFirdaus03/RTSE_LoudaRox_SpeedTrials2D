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

import perception
import policy
from policy import AgentState

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
    'acceleration_input' : 0.0,
    'agent_state': AgentState(),
}
data_lock = threading.Lock()
is_running = True

# Local-only buffers for simulating the yellow-token "delay" effects.
# These are touched by exactly one task each, so they need no lock.
CAMERA_DELAY_SECONDS = 0.5
CONTROL_DELAY_SECONDS = 0.5
_camera_delay_queue = deque(maxlen=200)
_control_delay_queue = deque(maxlen=200)

# Lane-tap pulse generator — owned exclusively by send_controls_task (the
# 200Hz loop, giving ~5ms timing precision). Reads state.lane_change_request
# (set by processing_task/policy.decide under data_lock) and drives
# target_steering through the documented "tap" gesture from the lab PDF:
# hold steering at +-1.0 for a short, calibrated wall-clock duration, then
# snap back to 0.0. Like _camera_delay_queue/_control_delay_queue, this dict
# is touched by exactly one task — no lock needed.
_tap = {
    'active': False,
    'direction': None,    # 'left' | 'right'
    'pulse_start': 0.0,
}

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
        except Exception as e:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
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
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

def processing_task():
    #This is where you write your image processing code to decide how to control the car
    #You can use libraries like OpenCV to process the image
    #There is no limtation to the complexity of the processing task, you can use any libraries you want
    #Remember to use the shared_data to get the latest frame
    now = time.time()
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']
        camera_delay_active = now < shared_data['agent_state'].camera_delay_until

    if front_frame is None:
        return

    # Simulate the "camera input delay" yellow-token effect: feed the
    # perception pipeline a frame from CAMERA_DELAY_SECONDS ago instead of
    # the freshest one. This deque is local to this task — no lock needed.
    if camera_delay_active:
        _camera_delay_queue.append((now, front_frame))
        ready = next((entry for entry in _camera_delay_queue
                      if now - entry[0] >= CAMERA_DELAY_SECONDS), None)
        if ready is not None:
            _camera_delay_queue.remove(ready)
        frame_to_process = ready[1] if ready is not None else front_frame
    else:
        _camera_delay_queue.clear()
        frame_to_process = front_frame

    # Heavy CV work runs outside the lock — these are pure functions over
    # local frame references (read_single_camera always assigns new arrays,
    # so it's safe to use these without holding data_lock).
    lane_info = perception.detect_current_lane(frame_to_process)
    tokens = perception.detect_tokens(frame_to_process)
    front_obstacles = perception.detect_front_obstacles(frame_to_process)
    rear = perception.detect_rear_events(back_frame)
    brightness = perception.measure_brightness(frame_to_process)

    with data_lock:
        state = shared_data['agent_state']
        acceleration = policy.decide(
            front_detections={
                'lane_index': lane_info['lane_index'],
                'lane_confidence': lane_info['confidence'],
                'lane_offset': lane_info['lane_offset'],
                'tokens': tokens,
                'obstacles': front_obstacles,
            },
            rear_detection=rear,
            brightness=brightness,
            state=state,
            now=now,
        )
        policy.update_effective_acceleration(state, acceleration)

def send_controls_task():
    #This is where you send the control commands to the car using the control_conn
    global control_conn
    if control_conn is None:
        return

    #these are the variables used to control the car
    #steering_input: -1.0 to 1.0 (left to right)
    #acceleration_input: -1.0 to 1.0 (reverse to forward)
    now = time.time()

    pulse_duration = policy.POLICY_CONFIG['lane_pulse_duration']
    settle_time = policy.POLICY_CONFIG['lane_change_settle_time']

    with data_lock:
        state = shared_data['agent_state']
        # Latch a fresh lane-change request into the local pulse generator.
        # Only one tap may be in flight at a time — _request_lane_change
        # already guarantees decide() won't issue overlapping requests, this
        # is just the handoff from "requested" to "executing".
        if (state.lane_change_request is not None
                and not _tap['active'] and not state.lane_change_in_progress):
            _tap['active'] = True
            _tap['direction'] = state.lane_change_request
            _tap['pulse_start'] = now
            state.lane_change_in_progress = True
            state.lane_change_request = None
        acceleration_input = state.target_acceleration
        control_delay_active = now < state.control_delay_until

    # --- Drive steering purely from the local tap generator ---
    # Pulse timing uses time.time() deltas (wall-clock), so the pulse width
    # is accurate to the lab PDF's "tap" gesture regardless of any jitter in
    # this loop's scheduling — this is the whole reason the tap executes here
    # (200Hz / ~5ms granularity) rather than in the ~30Hz processing_task.
    if _tap['active']:
        elapsed = now - _tap['pulse_start']
        if elapsed < pulse_duration:
            steering_input = 1.0 if _tap['direction'] == 'right' else -1.0
        else:
            steering_input = 0.0
            direction = _tap['direction']
            _tap['active'] = False
            _tap['direction'] = None
            with data_lock:
                state.lane_change_in_progress = False
                state.lane_change_settle_until = now + settle_time
                # Optimistic command-based lane update: gives decide() an
                # immediate belief about current_lane ahead of the next
                # confident vision reading, which then reconciles it
                # (perception._update_current_lane trusts vision once it's
                # confidently available — see policy._update_current_lane).
                if state.current_lane is not None:
                    delta = 1 if direction == 'right' else -1
                    state.current_lane = max(0, min(state.num_lanes - 1, state.current_lane + delta))
    else:
        steering_input = 0.0   # idle between taps: always neutral steering

    # Simulate the "action output delay" yellow-token effect: hold back the
    # freshest decision and send one that's CONTROL_DELAY_SECONDS old instead.
    if control_delay_active:
        _control_delay_queue.append((now, steering_input, acceleration_input))
        ready = [entry for entry in _control_delay_queue if now - entry[0] >= CONTROL_DELAY_SECONDS]
        if not ready:
            return
        _, steering_input, acceleration_input = ready[0]
        _control_delay_queue.remove(ready[0])
    else:
        _control_delay_queue.clear()

    try:
        # Pack and send the control command
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
    # Classical CV (Canny+Hough+multiple inRange+findContours) cannot keep up with
    # the 200Hz control loop, so perception runs at a slower ~30Hz cadence and writes
    # target_steering/target_acceleration into agent_state; send_controls_task keeps
    # transmitting at full rate from whatever values perception last computed.
    t_processing = RTTask("Processing", period=0.033, priority=TaskPriority.MEDIUM, execute_func=processing_task)
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

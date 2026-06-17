import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes
from agent_policy import HeuristicPolicy
from low_light import LowLightController
from police_car import PoliceCarController
from chasing_car import ChasingCarController

# The swappable "brain". To upgrade later, change this one line to a trained
# policy, e.g.  POLICY = LearnedPolicy("model.onnx")  -- nothing else changes.
POLICY = HeuristicPolicy()

# Challenge 1 (Low Light) handler -- a separate post-processing component that
# overrides acceleration to recover the light when the screen goes dark.
LOW_LIGHT = LowLightController()

# Challenge 3 (Police Car) handler -- a separate post-processing component that
# takes over steering to grab a red token and dodge the cop while it is on-screen.
POLICE = PoliceCarController()

# Challenge 2 (Chasing Car) handler -- watches the BACK camera and dodges the
# teal chaser when it closes in (swerve aside + full throttle).
CHASER = ChasingCarController()

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
    'acceleration_input' : 0.0
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

def recv_exact(sock, n):
    """Receive EXACTLY n bytes from a TCP socket.

    A single sock.recv(n) may return fewer than n bytes (TCP is a byte stream,
    not message-framed). Reading the 4-byte length prefix with a bare recv(4)
    can therefore under-read, yielding a wrong frame length, desyncing the
    stream, and producing truncated JPEGs that decode with a black bottom half.
    This loops until all n bytes arrive. Returns the bytes, or None if the
    connection closed / timed out before n bytes were received.
    """
    buf = bytearray()
    while len(buf) < n and is_running:
        try:
            packet = sock.recv(n - len(buf))
        except socket.timeout:
            return None
        if not packet:           # peer closed the connection
            return None
        buf += packet
    return bytes(buf) if len(buf) == n else None


def read_single_camera(sock, window_name, data_key):
    #This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return

    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = recv_exact(sock, 4)
        if not length_bytes:
            return

        image_length = int.from_bytes(length_bytes, 'little')
        latest_frame_data = recv_exact(sock, image_length)

        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break

            sock.settimeout(1.0)
            length_bytes = recv_exact(sock, 4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            frame_data = recv_exact(sock, image_length)
            if frame_data is not None:
                latest_frame_data = frame_data   # keep only the most recent full frame

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
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']

    if front_frame is not None:
        # Run the swappable policy: BGR frame in -> (steering, acceleration) out.
        steering, acceleration = POLICY.act(front_frame, back_frame)
        # Challenge 1 - Low Light: separate component overrides accel to recover the light.
        steering, acceleration = LOW_LIGHT.apply(front_frame, steering, acceleration)
        # Challenge 3 - Police Car: while the cop is on-screen, take over to grab a
        # red token and dodge the cop; otherwise pass the policy's output through.
        steering, acceleration = POLICE.apply(front_frame, steering, acceleration)
        # Challenge 2 - Chasing Car: runs LAST so its evasion overrides everything when active.
        steering, acceleration = CHASER.apply(back_frame, steering, acceleration)
        with data_lock:
            shared_data['steering_input'] = steering
            shared_data['acceleration_input'] = acceleration

def send_controls_task():
    #This is where you send the control commands to the car using the control_conn
    global control_conn
    if control_conn is None:
        return
    
    #these are the variables used to control the car
    #steering_input: -1.0 to 1.0 (left to right)
    #acceleration_input: -1.0 to 1.0 (reverse to forward)
    #Read the latest decision produced by processing_task() via shared_data.
    with data_lock:
        steering_input = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']

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

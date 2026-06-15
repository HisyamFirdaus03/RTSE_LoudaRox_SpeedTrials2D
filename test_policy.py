"""
Unit tests for policy.py's pure decision logic — no CV, no threading, no
locking. Constructs AgentState and synthetic detection namedtuples directly
and exercises _handle_front_obstacles and decide() in isolation.

Run with: python3 -m unittest test_policy.py -v
"""

import time
import unittest

import perception
import policy
from policy import AgentState, POLICY_CONFIG, _handle_front_obstacles, decide

EVADE = POLICY_CONFIG['front_obstacle_evade_distance']
BRAKE = POLICY_CONFIG['front_obstacle_brake_distance']


def make_obstacle(lane, distance, faster_car=True, police_car=False):
    return perception.ObstacleDetection(
        faster_car=faster_car, police_car=police_car, lane=lane,
        distance_estimate=distance, cx=0.0, cy=0.0, area=0.0,
    )


def make_state(current_lane=1, num_lanes=3, **overrides):
    state = AgentState()
    state.current_lane = current_lane
    state.lane_confidence = 1.0
    state.num_lanes = num_lanes
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def make_front_detections(lane_index, tokens=None, obstacles=None):
    return {
        'lane_index': lane_index,
        'lane_confidence': 1.0,
        'lane_offset': 0.0,
        'tokens': tokens or [],
        'obstacles': obstacles or [],
    }


class TestHandleFrontObstacles(unittest.TestCase):
    def test_no_obstacles_returns_none_false(self):
        state = make_state()
        self.assertEqual(_handle_front_obstacles(state, [], time.time()), (None, False))

    def test_obstacle_below_threshold_ignored(self):
        state = make_state(current_lane=1)
        obstacles = [make_obstacle(1, EVADE - 1)]
        self.assertEqual(_handle_front_obstacles(state, obstacles, time.time()), (None, False))

    def test_evades_to_clear_left_lane(self):
        state = make_state(current_lane=1)
        obstacles = [make_obstacle(1, EVADE + 100), make_obstacle(2, EVADE + 100)]
        direction, brake = _handle_front_obstacles(state, obstacles, time.time())
        self.assertEqual(direction, 'left')
        self.assertFalse(brake)

    def test_evades_to_clear_right_lane_when_left_blocked(self):
        state = make_state(current_lane=1)
        obstacles = [make_obstacle(0, EVADE + 100), make_obstacle(1, EVADE + 100)]
        direction, brake = _handle_front_obstacles(state, obstacles, time.time())
        self.assertEqual(direction, 'right')
        self.assertFalse(brake)

    def test_both_neighbors_blocked_brakes_when_close_enough(self):
        state = make_state(current_lane=1)
        obstacles = [
            make_obstacle(0, EVADE + 100),
            make_obstacle(1, BRAKE + 100),
            make_obstacle(2, EVADE + 100),
        ]
        direction, brake = _handle_front_obstacles(state, obstacles, time.time())
        self.assertIsNone(direction)
        self.assertTrue(brake)

    def test_both_neighbors_blocked_not_close_enough_to_brake(self):
        state = make_state(current_lane=1)
        ahead_distance = (EVADE + BRAKE) / 2.0
        obstacles = [
            make_obstacle(0, EVADE + 100),
            make_obstacle(1, ahead_distance),
            make_obstacle(2, EVADE + 100),
        ]
        direction, brake = _handle_front_obstacles(state, obstacles, time.time())
        self.assertIsNone(direction)
        self.assertFalse(brake)

    def test_unconfident_current_lane_returns_none(self):
        state = make_state(current_lane=None)
        obstacles = [make_obstacle(1, EVADE + 100)]
        self.assertEqual(_handle_front_obstacles(state, obstacles, time.time()), (None, False))

    def test_edge_lane_left_only_checks_right_neighbor(self):
        state = make_state(current_lane=0, num_lanes=3)
        obstacles = [make_obstacle(0, EVADE + 100)]  # lane 1 has no detection (clear)
        direction, brake = _handle_front_obstacles(state, obstacles, time.time())
        self.assertEqual(direction, 'right')
        self.assertFalse(brake)

    def test_edge_lane_left_blocked_brakes_without_checking_left(self):
        state = make_state(current_lane=0, num_lanes=3)
        obstacles = [make_obstacle(0, BRAKE + 100), make_obstacle(1, EVADE + 100)]
        direction, brake = _handle_front_obstacles(state, obstacles, time.time())
        self.assertIsNone(direction)
        self.assertTrue(brake)

    def test_neighbor_with_no_detection_treated_as_clear(self):
        state = make_state(current_lane=1)
        obstacles = [make_obstacle(1, EVADE + 100)]  # lanes 0 and 2 have no detections
        direction, brake = _handle_front_obstacles(state, obstacles, time.time())
        self.assertEqual(direction, 'left')
        self.assertFalse(brake)


class TestDecidePriorityOrdering(unittest.TestCase):
    def test_front_obstacle_wins_over_rear_event(self):
        state = make_state(current_lane=1, num_lanes=3)
        # Ahead blocked, left blocked, right clear -> front evades right.
        obstacles = [make_obstacle(0, EVADE + 100), make_obstacle(1, EVADE + 100)]
        # Faster car behind in our lane -> rear evasion would request left.
        rear = perception.RearDetection(faster_car=True, police_car=False, lane=1, distance_estimate=100.0)
        front_detections = make_front_detections(lane_index=1, obstacles=obstacles)

        decide(front_detections, rear, brightness=200.0, state=state, now=time.time())

        self.assertEqual(state.lane_change_request, 'right')

    def test_front_obstacle_wins_over_token_seeking(self):
        state = make_state(current_lane=1, num_lanes=3)
        # Ahead blocked, left and right clear -> front evades left (preferred).
        obstacles = [make_obstacle(1, EVADE + 100)]
        # A green token sits in lane 2 -> token-seeking would request right.
        token = perception.TokenDetection(color='green', lane=2, distance_estimate=100.0, cx=0.0, cy=0.0, area=0.0)
        front_detections = make_front_detections(lane_index=1, tokens=[token], obstacles=obstacles)

        decide(front_detections, None, brightness=200.0, state=state, now=time.time())

        self.assertEqual(state.lane_change_request, 'left')

    def test_rear_event_fires_without_front_obstacle(self):
        state = make_state(current_lane=1, num_lanes=3)
        rear = perception.RearDetection(faster_car=True, police_car=False, lane=1, distance_estimate=100.0)
        front_detections = make_front_detections(lane_index=1)

        decide(front_detections, rear, brightness=200.0, state=state, now=time.time())

        self.assertEqual(state.lane_change_request, 'left')

    def test_no_request_when_lane_change_in_progress(self):
        state = make_state(current_lane=1, num_lanes=3, lane_change_in_progress=True)
        obstacles = [make_obstacle(1, EVADE + 100)]
        front_detections = make_front_detections(lane_index=1, obstacles=obstacles)

        decide(front_detections, None, brightness=200.0, state=state, now=time.time())

        self.assertIsNone(state.lane_change_request)


class TestBrakingFallback(unittest.TestCase):
    def test_braking_fallback_overrides_acceleration(self):
        state = make_state(current_lane=1, num_lanes=3)
        obstacles = [
            make_obstacle(0, EVADE + 100),
            make_obstacle(1, BRAKE + 100),
            make_obstacle(2, EVADE + 100),
        ]
        front_detections = make_front_detections(lane_index=1, obstacles=obstacles)

        acceleration = decide(front_detections, None, brightness=200.0, state=state, now=time.time())

        self.assertEqual(acceleration, POLICY_CONFIG['front_obstacle_brake_acceleration'])
        self.assertLess(acceleration, POLICY_CONFIG['min_acceleration'])

    def test_no_braking_when_evasion_possible(self):
        state = make_state(current_lane=1, num_lanes=3)
        obstacles = [make_obstacle(1, BRAKE + 100)]  # very close, but lanes 0/2 are clear
        front_detections = make_front_detections(lane_index=1, obstacles=obstacles)

        acceleration = decide(front_detections, None, brightness=200.0, state=state, now=time.time())

        self.assertEqual(state.lane_change_request, 'left')
        self.assertEqual(acceleration, 1.0)

    def test_braking_overrides_collision_penalty(self):
        now = time.time()
        state = make_state(current_lane=1, num_lanes=3, collision_penalty_until=now + 10.0)
        obstacles = [
            make_obstacle(0, EVADE + 100),
            make_obstacle(1, BRAKE + 100),
            make_obstacle(2, EVADE + 100),
        ]
        front_detections = make_front_detections(lane_index=1, obstacles=obstacles)

        acceleration = decide(front_detections, None, brightness=200.0, state=state, now=now)

        self.assertEqual(acceleration, POLICY_CONFIG['front_obstacle_brake_acceleration'])


class TestDetectFrontObstacles(unittest.TestCase):
    def test_none_frame_returns_empty_list(self):
        self.assertEqual(perception.detect_front_obstacles(None), [])


if __name__ == '__main__':
    unittest.main()

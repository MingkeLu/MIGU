#!/usr/bin/env python3
"""Detect stable pointing episodes in a ROS RGB-D stream."""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import yaml

try:
    from .key_frame2ray import (
        CameraIntrinsics, depth_at_pixel_robust, estimate_index_finger_ray, unproject,
    )
except ImportError:
    from key_frame2ray import (
        CameraIntrinsics, depth_at_pixel_robust, estimate_index_finger_ray, unproject,
    )


RGB_TOPIC = "/head_camera/rgb/image_raw"
INFO_TOPIC = "/head_camera/rgb/camera_info"
DEPTH_TOPIC = "/head_camera/depth_registered/image"
DEPTH_SCALE_METRES = 0.001
DEPTH_MATCH_SECONDS = 0.05
STATIC_SPEED_THRESHOLD = 80.0
MOVING_SPEED_THRESHOLD = 60.0
SPEED_EMA_ALPHA = 0.5
ENTER_STATIC_FRAMES = 1
ENTER_STRAIGHT_FRAMES = 2
EXIT_MOVING_FRAMES = 2
EXIT_IDLE_FRAMES = 4


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("grounding_inputs"))
    parser.add_argument("--save-keyframes", action="store_true")
    return parser.parse_known_args(argv)[0]


def _native_array(message: Image, dtype: np.dtype, channels: int) -> np.ndarray:
    """Decode a tightly or row-padded ROS Image without cv_bridge."""
    dtype = np.dtype(dtype).newbyteorder(">" if message.is_bigendian else "<")
    row_items = message.step // dtype.itemsize
    values = np.frombuffer(message.data, dtype=dtype).reshape(message.height, row_items)
    values = values[:, : message.width * channels]
    shape = ((message.height, message.width) if channels == 1
             else (message.height, message.width, channels))
    return np.asarray(values.reshape(shape), dtype=dtype.newbyteorder("="))


def image_to_bgr(message: Image) -> np.ndarray:
    """Convert common ROS color encodings to an OpenCV BGR array."""
    encoding = message.encoding.lower()
    if encoding == "bgr8":
        return _native_array(message, np.uint8, 3).copy()
    if encoding == "rgb8":
        return cv2.cvtColor(_native_array(message, np.uint8, 3), cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(_native_array(message, np.uint8, 4), cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(_native_array(message, np.uint8, 4), cv2.COLOR_RGBA2BGR)
    if encoding in {"mono8", "8uc1"}:
        return cv2.cvtColor(_native_array(message, np.uint8, 1), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported RGB encoding: {message.encoding}")


def image_to_depth(message: Image) -> tuple[np.ndarray, float]:
    """Decode a ROS depth image and return its array and metres-per-unit scale."""
    encoding = message.encoding.lower()
    if encoding in {"16uc1", "mono16"}:
        return _native_array(message, np.uint16, 1).copy(), DEPTH_SCALE_METRES
    if encoding == "32fc1":
        return _native_array(message, np.float32, 1).copy(), 1.0
    raise ValueError(f"Unsupported depth encoding: {message.encoding}")


def load_intrinsics(path: Path) -> tuple[float, float, float, float]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        if key.strip() in {"fx", "fy", "cx", "cy"}:
            values[key.strip()] = float(value.strip())
    missing = {"fx", "fy", "cx", "cy"} - values.keys()
    if missing:
        raise ValueError(f"Missing intrinsics in {path}: {sorted(missing)}")
    return values["fx"], values["fy"], values["cx"], values["cy"]


class GestureMonitor:
    """Watch RGB-D frames and publish one payload for each pointing episode."""

    def __init__(self, args):
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        hand = config["hand_detection"]
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=float(hand["min_detection_confidence"]),
            min_tracking_confidence=float(hand["min_tracking_confidence"]),
        )
        self.intrinsics = load_intrinsics(args.intrinsics) if args.intrinsics else None
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_keyframes = args.save_keyframes
        self.depth_buffer = deque(maxlen=150)
        self.lock = threading.Lock()
        self.previous_finger = None
        self.previous_time = None
        self.speed_ema = 0.0
        self.latest_landmarks = None
        self.phase = "detecting"
        self.enter_static = 0
        self.enter_straight = 0
        self.exit_moving = 0
        self.exit_idle = 0
        self.capture_index = self._next_capture_index()
        self.segment_index = 1
        self.segment_last_time = None
        self.segment_rgb_frames = 0
        self.segment_hand_frames = 0
        self.segment_straight_frames = 0
        self.segment_max_enter_streak = 0
        self.segment_triggers = 0
        self.segment_straight_speeds = []

        self.query_publisher = rospy.Publisher("/ray_queries", String, queue_size=10)
        self.marker_publisher = rospy.Publisher("/ray_marker", Marker, queue_size=10)
        rospy.Subscriber(RGB_TOPIC, Image, self.on_rgb, queue_size=5, buff_size=2**24)
        rospy.Subscriber(DEPTH_TOPIC, Image, self.on_depth, queue_size=5, buff_size=2**24)
        if self.intrinsics is None:
            rospy.Subscriber(INFO_TOPIC, CameraInfo, self.on_camera_info, queue_size=1)
        rospy.loginfo("gesture_monitor ready; output=%s", self.output_dir)

    def close(self):
        self._log_segment_summary()
        self.hands.close()

    def _log_segment_summary(self):
        if not self.segment_rgb_frames:
            return
        speeds = np.asarray(self.segment_straight_speeds, dtype=float)
        speed_summary = ("none" if not len(speeds) else
                         "min=%.1f p10=%.1f median=%.1f" % tuple(
                             np.percentile(speeds, [0, 10, 50])))
        rospy.loginfo(
            "Gesture segment %d summary: rgb=%d hand=%d straight=%d "
            "max_enter_streak=%d triggers=%d straight_speed=(%s)",
            self.segment_index, self.segment_rgb_frames, self.segment_hand_frames,
            self.segment_straight_frames, self.segment_max_enter_streak,
            self.segment_triggers, speed_summary,
        )

    def _start_new_segment(self):
        self._log_segment_summary()
        self.segment_index += 1
        self.segment_rgb_frames = 0
        self.segment_hand_frames = 0
        self.segment_straight_frames = 0
        self.segment_max_enter_streak = 0
        self.segment_triggers = 0
        self.segment_straight_speeds = []
        self.previous_finger = None
        self.previous_time = None
        self.speed_ema = 0.0
        self.phase = "detecting"
        self.enter_static = self.enter_straight = 0
        self.exit_moving = self.exit_idle = 0

    def _next_capture_index(self) -> int:
        values = []
        for path in self.output_dir.glob("*_rgb.png"):
            try:
                values.append(int(path.name.split("_", 1)[0]))
            except ValueError:
                pass
        return max(values, default=0) + 1

    def on_camera_info(self, message: CameraInfo):
        if self.intrinsics is None:
            self.intrinsics = (message.K[0], message.K[4], message.K[2], message.K[5])
            rospy.loginfo("Loaded intrinsics from %s", INFO_TOPIC)

    @staticmethod
    def _straight_index(landmarks, width: int, height: int) -> bool:
        points = np.asarray(
            [[landmarks[index].x * width, landmarks[index].y * height]
             for index in (5, 6, 7, 8)], dtype=np.float32,
        )
        x, y = points[:, 0], points[:, 1]
        if np.std(x) < 1e-3:
            x, y = y, x
        design = np.vstack([x, np.ones_like(x)]).T
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        r_squared = 1.0 - np.square(y - fitted).sum() / (np.square(y - y.mean()).sum() + 1e-9)
        vectors = np.diff(points, axis=0)
        angle_variance = np.var([math.degrees(math.atan2(v[1], v[0] + 1e-9))
                                 for v in vectors])
        return bool(r_squared >= 0.98 and angle_variance <= 15.0)

    def on_rgb(self, message: Image):
        if self.intrinsics is None:
            return
        try:
            bgr = image_to_bgr(message)
        except ValueError as error:
            rospy.logerr_throttle(5.0, str(error))
            return
        timestamp = message.header.stamp.to_sec()
        if self.segment_last_time is not None and timestamp - self.segment_last_time > 2.0:
            self._start_new_segment()
        self.segment_last_time = timestamp
        self.segment_rgb_frames += 1
        result = self.hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finger = None
        straight = False
        if result.multi_hand_landmarks:
            self.segment_hand_frames += 1
            landmarks = result.multi_hand_landmarks[0].landmark
            height, width = bgr.shape[:2]
            finger = np.asarray(
                [[landmarks[index].x * width, landmarks[index].y * height]
                 for index in (5, 6, 7, 8)], dtype=np.float32,
            )
            straight = self._straight_index(landmarks, width, height)
            if straight:
                self.segment_straight_frames += 1
            self.latest_landmarks = [
                (int(point.x * width), int(point.y * height)) for point in landmarks
            ]

        instantaneous = None
        if finger is not None and self.previous_finger is not None and self.previous_time is not None:
            elapsed = timestamp - self.previous_time
            if elapsed > 1e-6:
                instantaneous = float(np.median(
                    np.linalg.norm(finger - self.previous_finger, axis=1) / elapsed
                ))
        self.speed_ema = ((SPEED_EMA_ALPHA * instantaneous
                           + (1.0 - SPEED_EMA_ALPHA) * self.speed_ema)
                          if instantaneous is not None else self.speed_ema * 0.95)
        self.previous_finger = finger
        if finger is not None:
            self.previous_time = timestamp
        is_static = finger is not None and self.speed_ema <= STATIC_SPEED_THRESHOLD
        is_moving = finger is not None and self.speed_ema >= MOVING_SPEED_THRESHOLD
        if finger is not None and straight:
            self.segment_straight_speeds.append(self.speed_ema)

        if self.phase == "detecting":
            self.enter_static = self.enter_static + 1 if is_static else 0
            self.enter_straight = self.enter_straight + 1 if straight else 0
            self.segment_max_enter_streak = max(
                self.segment_max_enter_streak,
                min(self.enter_static, self.enter_straight),
            )
            if (self.enter_static >= ENTER_STATIC_FRAMES
                    and self.enter_straight >= ENTER_STRAIGHT_FRAMES):
                if self.capture(timestamp, bgr):
                    self.segment_triggers += 1
                    rospy.loginfo("POINTING detected at bag time %.6f", timestamp)
                    self.phase = "wait_exit"
                self.enter_static = self.enter_straight = 0
        else:
            self.exit_moving = self.exit_moving + 1 if is_moving else 0
            self.exit_idle = self.exit_idle + 1 if not straight else 0
            if self.exit_moving >= EXIT_MOVING_FRAMES or self.exit_idle >= EXIT_IDLE_FRAMES:
                self.phase = "detecting"
                self.exit_moving = self.exit_idle = 0

    def on_depth(self, message: Image):
        try:
            depth, scale = image_to_depth(message)
        except ValueError as error:
            rospy.logerr_throttle(5.0, str(error))
            return
        with self.lock:
            self.depth_buffer.append((message.header.stamp.to_sec(), depth, scale))

    def capture(self, rgb_time: float, bgr: np.ndarray) -> bool:
        with self.lock:
            if not self.depth_buffer:
                rospy.logwarn("No depth frame is available for the pointing trigger")
                return False
            candidate = min(self.depth_buffer, key=lambda item: abs(item[0] - rgb_time))
        depth_time, depth, scale = candidate
        delta = abs(depth_time - rgb_time)
        if delta > DEPTH_MATCH_SECONDS * 2:
            rospy.logwarn("Closest depth frame is %.3f seconds from RGB", delta)
            return False
        return self.publish_query(bgr, depth, scale, rgb_time, depth_time, delta)

    def publish_query(self, bgr, depth, scale, rgb_time, depth_time, delta) -> bool:
        if self.latest_landmarks is None or len(self.latest_landmarks) != 21:
            return False
        fx, fy, cx, cy = self.intrinsics
        intrinsics = CameraIntrinsics(fx, fy, cx, cy, depth_scale=1.0 / scale)
        ray = estimate_index_finger_ray(self.latest_landmarks, depth, intrinsics)
        if ray is None:
            rospy.logwarn("Pointing trigger has insufficient finger depth")
            return False
        origin, direction, used_points = ray
        keypoints = {}
        for index, (u, v) in enumerate(self.latest_landmarks):
            z = depth_at_pixel_robust(depth, u, v, depth_scale=intrinsics.depth_scale)
            if z is not None:
                keypoints[str(index)] = unproject(u, v, z, intrinsics).tolist()
        if not {"5", "6", "7", "8"}.issubset(keypoints):
            rospy.logwarn("Pointing trigger is missing one or more index-finger depths")
            return False

        capture_id = f"{self.capture_index:05d}"
        rgb_path = self.output_dir / f"{capture_id}_rgb.png"
        depth_path = self.output_dir / f"{capture_id}_depth.npz"
        if not cv2.imwrite(str(rgb_path), bgr):
            raise IOError(f"Could not write {rgb_path}")
        np.savez_compressed(depth_path, depth=depth)
        self.capture_index += 1
        payload = {
            "capture_id": capture_id,
            "origin": origin.tolist(),
            "direction": direction.tolist(),
            "timestamp": time.time(),
            "timestamp_rgb": rgb_time,
            "timestamp_depth": depth_time,
            "depth_match_dt": delta,
            "grounding": {
                "rgb": str(rgb_path),
                "depth": str(depth_path),
                "depth_scale": scale,
                "hand_keypoints": keypoints,
                "hand_landmarks_px": self.latest_landmarks,
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            },
        }
        self.query_publisher.publish(String(data=json.dumps(payload)))
        self._publish_marker(origin, direction)
        rospy.loginfo(
            "Published /ray_queries capture=%s points=%d rgb=%s",
            capture_id, len(used_points), rgb_path,
        )
        return True

    def _publish_marker(self, origin, direction):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "head_camera_rgb_optical_frame"
        marker.ns = "finger_ray"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        endpoint = origin + direction * 0.5
        marker.points = [Point(*map(float, origin)), Point(*map(float, endpoint))]
        marker.scale.x, marker.scale.y, marker.scale.z = 0.01, 0.02, 0.02
        marker.color.r, marker.color.a = 1.0, 1.0
        self.marker_publisher.publish(marker)


def main():
    args = parse_arguments(rospy.myargv(sys.argv)[1:])
    rospy.init_node("gesture_monitor")
    monitor = GestureMonitor(args)
    try:
        rospy.spin()
    finally:
        monitor.close()


if __name__ == "__main__":
    main()

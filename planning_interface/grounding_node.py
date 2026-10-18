#!/usr/bin/env python3
"""ROS node that combines user instructions with pointing-trigger payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import threading

import rospy
from std_msgs.msg import String

from .grounding_tool import GroundingTool, save_result


class GroundingNode:
    """Consume `/usr_instruction` and `/ray_queries`, then publish MIGU results."""

    def __init__(self, config: Path, output_dir: Path):
        self.tool = GroundingTool(config)
        self.output_dir = output_dir.resolve()
        self.instruction = None
        self.instruction_lock = threading.Lock()
        self.queries = queue.Queue()
        self.publisher = rospy.Publisher("/grounding_results", String, queue_size=10)
        rospy.Subscriber("/usr_instruction", String, self.on_instruction, queue_size=10)
        rospy.Subscriber("/ray_queries", String, self.on_query, queue_size=10)
        self.worker = threading.Thread(target=self.run_worker, daemon=True)
        self.worker.start()
        rospy.loginfo("grounding_node ready; results=%s", self.output_dir)

    def on_instruction(self, message: String):
        value = message.data.strip()
        if not value:
            rospy.logwarn("Ignoring an empty /usr_instruction message")
            return
        with self.instruction_lock:
            self.instruction = value
        rospy.loginfo("Received user instruction: %s", value)

    def on_query(self, message: String):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as error:
            rospy.logerr("Invalid /ray_queries JSON: %s", error)
            return
        with self.instruction_lock:
            instruction = self.instruction
        if instruction is None:
            rospy.logwarn("Ignoring pointing trigger until /usr_instruction is received")
            return
        self.queries.put((instruction, payload))
        rospy.loginfo("Queued grounding capture %s", payload.get("capture_id", "unknown"))

    def run_worker(self):
        while not rospy.is_shutdown():
            try:
                instruction, payload = self.queries.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                result = self.tool(instruction, payload)
                result_path = save_result(result, self.output_dir)
                self.publisher.publish(String(data=json.dumps(result)))
                rospy.loginfo(
                    "Grounding complete: capture=%s candidate=%d result=%s",
                    result.get("capture_id"), result["selected_index"] + 1, result_path,
                )
            except Exception as error:
                record = {
                    "status": "error",
                    "capture_id": payload.get("capture_id"),
                    "instruction": instruction,
                    "error": f"{type(error).__name__}: {error}",
                }
                save_result(record, self.output_dir)
                self.publisher.publish(String(data=json.dumps(record)))
                rospy.logerr("Grounding failed: %s", record["error"])
            finally:
                self.queries.task_done()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(rospy.myargv()[1:])
    rospy.init_node("grounding_node")
    GroundingNode(args.config, args.output_dir)
    rospy.spin()


if __name__ == "__main__":
    main()

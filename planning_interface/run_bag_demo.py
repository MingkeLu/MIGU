#!/usr/bin/env python3
"""Launch local SAM, the MIGU ROS interface, and one or more ROS bags."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS_BIN = Path("/opt/ros/noetic/bin")
ROS_PYTHON = Path("/opt/ros/noetic/lib/python3/dist-packages")


def _wait_health(url: str, timeout: float, process: subprocess.Popen | None = None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        result = subprocess.run(
            ["curl", "-fsS", "--max-time", "2", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
        time.sleep(2)
    return False


def _terminate(processes: list[subprocess.Popen]):
    for process in reversed(processes):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 8
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return "log file was not created"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--instruction", default="pick up this")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--intrinsics", type=Path,
                        default=PROJECT_ROOT / "bags" / "camera_intrinsics.txt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sam-model", default="facebook/sam-vit-huge")
    parser.add_argument("--sam-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sam-start-timeout", type=float, default=1800)
    parser.add_argument("--result-timeout", type=float, default=900)
    parser.add_argument("--expected-results", type=int)
    parser.add_argument("--detect-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = args.config.resolve()
    intrinsics = args.intrinsics.resolve()
    bags = [path.resolve() for path in args.bags]
    missing = [path for path in [config, intrinsics, *bags] if not path.is_file()]
    if missing:
        parser.error("missing required files: " + ", ".join(map(str, missing)))
    output_dir = (args.output_dir.resolve() if args.output_dir else
                  PROJECT_ROOT / "output" / "planning_interface" /
                  datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    inputs_dir = output_dir / "grounding_inputs"
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    for directory in (inputs_dir, results_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    ros_home = output_dir / "ros_home"
    ros_log_dir = logs_dir / "ros"
    ros_home.mkdir(parents=True, exist_ok=True)
    ros_log_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    ros_prefix = Path("/opt/ros/noetic")
    environment.setdefault("ROS_VERSION", "1")
    environment.setdefault("ROS_PYTHON_VERSION", "3")
    environment.setdefault("ROS_DISTRO", "noetic")
    environment.setdefault("ROS_MASTER_URI", "http://localhost:11311")
    environment.setdefault("ROS_ROOT", str(ros_prefix / "share" / "ros"))
    environment.setdefault("ROS_ETC_DIR", str(ros_prefix / "etc" / "ros"))
    environment.setdefault("ROS_PACKAGE_PATH", str(ros_prefix / "share"))
    environment["PATH"] = str(ros_prefix / "bin") + os.pathsep + environment["PATH"]
    environment["LD_LIBRARY_PATH"] = str(ros_prefix / "lib") + os.pathsep + environment.get(
        "LD_LIBRARY_PATH", ""
    )
    python_path = [str(PROJECT_ROOT), str(ROS_PYTHON)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment["MIGU_SAM_SERVER_URL"] = "http://127.0.0.1:4001"
    environment["ROS_HOME"] = str(ros_home)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)
    environment.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    if args.runtime_config:
        runtime = yaml.safe_load(args.runtime_config.resolve().read_text(encoding="utf-8"))
        vlm = runtime.get("vlm", {})
        if vlm.get("base_url"):
            environment["OPENAI_BASE_URL"] = str(vlm["base_url"])
        if vlm.get("api_key"):
            environment["OPENAI_API_KEY"] = str(vlm["api_key"])
    if not environment.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set")

    processes: list[subprocess.Popen] = []
    logs = []

    def handle_stop(signum, frame):
        del signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    def start(name: str, command: list[str]) -> subprocess.Popen:
        log_path = logs_dir / f"{name}.log"
        handle = log_path.open("w", encoding="utf-8")
        logs.append(handle)
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        processes.append(process)
        return process

    try:
        roscore = start("roscore", [str(ROS_BIN / "roscore")])
        time.sleep(3)
        if roscore.poll() is not None:
            roscore_output = _tail(logs_dir / "roscore.log")
            if "already running" not in roscore_output:
                raise RuntimeError("roscore exited during startup:\n" + roscore_output)

        sam = None
        if not args.detect_only and not _wait_health("http://127.0.0.1:4001/health", 1):
            sam = start("sam_server", [
                sys.executable, "-m", "migu.sam_server",
                "--host", "127.0.0.1", "--port", "4001",
                "--model", args.sam_model, "--device", args.sam_device,
            ])
            print(f"Loading local SAM model {args.sam_model}; first launch may download weights...")
            if not _wait_health(
                "http://127.0.0.1:4001/health", args.sam_start_timeout, sam
            ):
                raise RuntimeError("SAM did not become ready:\n" + _tail(logs_dir / "sam_server.log"))

        grounding = None
        if not args.detect_only:
            grounding = start("grounding_node", [
                sys.executable, "-m", "planning_interface.grounding_node",
                "--config", str(config), "--output-dir", str(results_dir),
            ])
        monitor = start("gesture_monitor", [
            sys.executable, "-m", "planning_interface.gesture_monitor",
            "--config", str(config), "--intrinsics", str(intrinsics),
            "--output-dir", str(inputs_dir),
        ])
        time.sleep(3)
        if grounding is not None and grounding.poll() is not None:
            raise RuntimeError("grounding_node exited:\n" + _tail(logs_dir / "grounding_node.log"))
        if monitor.poll() is not None:
            raise RuntimeError("gesture_monitor exited:\n" + _tail(logs_dir / "gesture_monitor.log"))

        if not args.detect_only:
            publisher = start("usr_instruction_publisher", [
                sys.executable, "-m", "planning_interface.usr_instruction_publisher",
                "--text", args.instruction,
            ])
            if publisher.wait(timeout=15) != 0:
                raise RuntimeError("instruction publisher failed:\n" +
                                   _tail(logs_dir / "usr_instruction_publisher.log"))
        bag_process = start("rosbag", [str(ROS_BIN / "rosbag"), "play", *map(str, bags)])
        if bag_process.wait() != 0:
            raise RuntimeError("rosbag play failed:\n" + _tail(logs_dir / "rosbag.log"))
        if args.detect_only:
            time.sleep(2)
            print(f"Detection-only run captured {len(list(inputs_dir.glob('*_rgb.png')))} "
                  f"pointing event(s): {inputs_dir}")
            return

        detected = len(list(inputs_dir.glob("*_rgb.png")))
        expected = args.expected_results if args.expected_results is not None else detected
        if expected == 0:
            print(f"No pointing event was detected; see {logs_dir / 'gesture_monitor.log'}")
            return
        deadline = time.monotonic() + args.result_timeout
        while time.monotonic() < deadline:
            records = sorted(results_dir.glob("*.json"))
            complete = []
            for path in records:
                try:
                    complete.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    pass
            if len(complete) >= expected:
                errors = [item for item in complete if item.get("status") != "ok"]
                if errors:
                    raise RuntimeError(f"{len(errors)} grounding request(s) failed; see {results_dir}")
                print(f"Completed {len(complete)} grounding result(s): {results_dir}")
                return
            if grounding.poll() is not None or monitor.poll() is not None:
                raise RuntimeError("A ROS interface process exited before results completed")
            time.sleep(2)
        raise TimeoutError(
            f"Expected {expected} results but found {len(list(results_dir.glob('*.json')))}; "
            f"see logs in {logs_dir}"
        )
    finally:
        _terminate(processes)
        for handle in logs:
            handle.close()


if __name__ == "__main__":
    main()

"""Run nine MIGU ablations through shared inference and analysis modules."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path

from openai import OpenAI
import yaml

from baselines.runner import write_json
from batch_preparation.shared import DEFAULT_CONFIG, PROJECT_ROOT, select_groups
from migu.datasets import open_dataset, pair_prepared_frames
from migu.key_frames import generate_key_frames
from migu.pipeline import load_config, process_frame
from .semantic import BASELINE_NAMES, find_snapshot, make_runner

GEOMETRY_MODES = {"ray-distance": "distance_cone", "ray-angle": "angle_error_gaussian",
                  "distance-threshold": "dist_threshold"}
VARIANTS = ("without-semantics", "without-geometry", *GEOMETRY_MODES,
            "semantic-owlv2", "semantic-som", "semantic-grid", "semantic-pivot")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_CONFIG,
                        help="Existing segmentation vocabulary/service and baseline method options")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--subdirs", nargs="*")
    parser.add_argument("--output-dir", type=Path, help="Ablation output root; a timestamp is appended")
    parser.add_argument("--batch", action="store_true",
                        help="Read normalized inputs/rays and baseline masks from the prepared cache")
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--reuse-result-dir", type=Path,
                        help="Reuse MIGU segmentation or the replacement baseline's semantic results")
    parser.add_argument("--analyze", "--analyse", action="store_true")
    parser.add_argument("--analysis-input-dir", type=Path,
                        help="Root containing <group>/ground_truth_masks")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--owlv2-server-url", help="Override the existing OWL service URL")
    parser.add_argument("--owlv2-bbox-conf-threshold", type=float, default=0.15)
    parser.add_argument("--owlv2-confidence", type=float, default=1.0,
                        help="Uniform candidate confidence, matching the original OWLv2 ablation")
    return parser


def main():
    args = build_parser().parse_args()
    if args.reuse_result_dir is not None:
        if args.variant in {"without-semantics", "semantic-owlv2"}:
            raise SystemExit("This variant uses its configured detector; reuse prepared masks with --batch for without-semantics")
        args.reuse_result_dir = args.reuse_result_dir.resolve()
        if not args.reuse_result_dir.is_dir():
            raise SystemExit(f"Reuse directory not found: {args.reuse_result_dir}")
    if not 0 <= args.owlv2_bbox_conf_threshold <= 1 or not 0 <= args.owlv2_confidence <= 1:
        raise SystemExit("OWLv2 threshold and confidence must lie in [0, 1]")
    config = load_config(args.config.resolve())
    config["gesture"]["mode"] = GEOMETRY_MODES.get(args.variant, config["gesture"]["mode"])
    visualize = config.get("visualize", True) if args.visualize is None else args.visualize
    source = (args.input_dir or PROJECT_ROOT / config["input_dir"]).resolve()
    cache_root = (args.prepared_dir or PROJECT_ROOT / "batch_preparation" / source.name / "input").resolve()
    output_root = (args.output_dir or PROJECT_ROOT / "output" / "ablations" / args.variant).resolve()
    analysis_input = args.analysis_input_dir or (
        cache_root if any(cache_root.glob("*/ground_truth_masks/*.npz")) else source)
    if args.analyze and not any(analysis_input.glob("*/ground_truth_masks/*.npz")):
        raise FileNotFoundError(
            "Automatic analysis requires <group>/ground_truth_masks/*.npz; "
            "run python -m batch_preparation first or pass --analysis-input-dir")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = output_root / stamp
    baseline_config = None
    if args.variant == "without-semantics" or args.variant.startswith("semantic-"):
        baseline_config = deepcopy(yaml.safe_load(args.baseline_config.read_text()))
        baseline_config["vlm"] = deepcopy(config["vlm"])
        baseline_config["visualize"] = visualize
    client = None
    needs_client = (args.reuse_result_dir is None and
                    (args.variant in GEOMETRY_MODES or args.variant in {"without-geometry", "semantic-owlv2"}))
    if needs_client:
        client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"),
                        api_key=os.environ.get("OPENAI_API_KEY"))
    summaries = []
    for group, group_source in select_groups(source, args.subdirs if args.subdirs is not None else config.get("subdirs")):
        group_run = run / group
        if args.batch:
            cache = cache_root / group
            if not cache.is_dir():
                raise FileNotFoundError(f"Missing prepared group: {cache}; run python -m batch_preparation first")
            prepared = open_dataset(cache).prepare(cache)
        else:
            prepared = open_dataset(group_source).prepare(group_run / "_prepared_input")
        hand_stats = {"saved": 0, "disabled": args.variant == "without-geometry",
                      "processing_time_seconds_by_frame": {}}
        if args.variant != "without-geometry" and not args.batch:
            hand_stats = generate_key_frames(prepared.rgb_dir, prepared.depth_dir, prepared.ray_dir,
                                             prepared.intrinsics_path, config["hand_detection"])
        frames = pair_prepared_frames(prepared, require_gesture=False)
        if not frames:
            raise ValueError(f"No RGB frames found for {group}")
        if args.reuse_result_dir is not None:
            for frame in frames:
                find_snapshot(args.reuse_result_dir, group, frame.frame_id, args.variant)
        if args.batch and args.variant != "without-geometry":
            manifest_path = cache_root / group / "manifest.json"
            manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.is_file() else {}
            hand_stats["processing_time_seconds_by_frame"] = {
                fid: timing.get("hand_detection_seconds")
                for fid, timing in manifest.get("frame_timings", {}).items()}
            hand_stats["saved"] = sum(frame.ray_path is not None for frame in frames)
            hand_stats["reused_from"] = str(cache_root / group)
        semantic_runner = None
        if baseline_config is not None or args.reuse_result_dir is not None:
            semantic_runner = make_runner(args.variant, group, group_run, baseline_config, args, cache_root)
        mode = ("geometry" if args.variant == "without-semantics" else
                "semantic" if args.variant == "without-geometry" else "fusion")
        results = []
        for frame in frames:
            instruction = prepared.instruction_by_frame.get(frame.frame_id)
            if args.variant == "without-semantics":
                instruction = instruction or ""
            elif not instruction:
                raise ValueError(f"No instruction for {group}/{frame.frame_id}")
            result = process_frame(
                frame, instruction, prepared.intrinsics_path, group_run, config, client,
                visualize, None if mode == "semantic" else
                hand_stats["processing_time_seconds_by_frame"].get(frame.frame_id),
                semantic_runner=semantic_runner, scoring_mode=mode)
            result["ablation"] = args.variant
            write_json(group_run / frame.frame_id / "result.json", result)
            results.append(result)
        summary = {"group": group, "ablation": args.variant,
                   "hand_detection": hand_stats, "frames": results}
        write_json(group_run / "summary.json", summary)
        summaries.append(summary)
    write_json(run / "summary.json", {"created_at": stamp, "ablation": args.variant,
                                      "gesture_mode": config["gesture"]["mode"], "groups": summaries})
    if args.analyze:
        from migu.result_analysis import analyse_migu_run
        report = analyse_migu_run(run, analysis_input)
        print(f"Accuracy: {report['accuracy']}")
    print(f"Output: {run}")


if __name__ == "__main__":
    main()

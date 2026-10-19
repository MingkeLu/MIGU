"""Run public baseline methods using one data and configuration interface."""
from __future__ import annotations
import argparse
from dataclasses import fields
from datetime import datetime
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import yaml
from migu.datasets import open_dataset, pair_prepared_frames, select_capture_groups
from batch_preparation.shared import (
    DEFAULT_CONFIG, PROJECT_ROOT, prepare_group, segmentation_config_for_group,
    load_perception, segment_image, select_groups,
)
from batch_preparation.perception import overlay_results

METHODS = ("geometry", "POMDP", "GIRAF", "VLM-Grid", "VLM-PIVOT", "VLM-SoM")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_parser(default_method=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if default_method is None:
        parser.add_argument("method", choices=METHODS)
    else:
        parser.set_defaults(method=default_method)
    parser.add_argument("image", nargs="?", type=Path,
                        help="Optional single RGB image for VLM methods")
    parser.add_argument("--instruction", help="Override every dataset instruction")
    parser.add_argument("--group", help="Vocabulary group for a single RGB image")
    parser.add_argument("--vocabulary", nargs="+", help="Override segmentation vocabulary")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path, help="Prepared cache root containing groups")
    parser.add_argument("--subdirs", nargs="*")
    parser.add_argument("--output-dir", type=Path, help="Method output root; a timestamp is appended")
    parser.add_argument("--batch", action="store_true", help="Reuse prepared rays and masks")
    parser.add_argument("--analyze", "--analyse", action="store_true",
                        help="Generate accuracy, localization, calibration, and timing reports")
    parser.add_argument("--language-provider", choices=("som", "embedding", "hashing"))
    parser.add_argument("--semantic-provider", choices=("openai", "lexical"))
    visual = parser.add_mutually_exclusive_group()
    visual.add_argument("--visualize", dest="visualize", action="store_true")
    visual.add_argument("--no-visualize", dest="visualize", action="store_false")
    parser.set_defaults(visualize=None)
    return parser


def method_context(method, config, prepared, segmentation, args, cache):
    options = config["methods"][method]
    context = SimpleNamespace(batch=True, batch_input=cache, segmentation=segmentation,
                              visualize=args.visualize, base_url=os.environ.get("OPENAI_BASE_URL"),
                              api_key=os.environ.get("OPENAI_API_KEY"))
    if method == "POMDP":
        from .POMDP.implementation import load_intrinsics
        from .POMDP.model import (POMDPConfig, EmbeddingLanguageSimilarityModel,
                                  OpenAIEmbeddingBackend, hashing_embedding_backend)
        context.intrinsics = load_intrinsics(prepared.intrinsics_path)
        context.model_config = POMDPConfig(**options["model"])
        language = options["language"]
        context.language_provider = args.language_provider or language["provider"]
        context.language_model_name = config["vlm"]["model"]
        context.language_base_url, context.language_api_key = context.base_url, context.api_key
        context.language_model = None
        if context.language_provider == "hashing":
            context.language_model = EmbeddingLanguageSimilarityModel(hashing_embedding_backend)
        elif context.language_provider == "embedding":
            backend = OpenAIEmbeddingBackend(
                model=language.get("embedding_model", "text-embedding-3-small"),
                base_url=context.base_url, api_key=context.api_key,
                dimensions=language.get("dimensions"))
            context.language_model = EmbeddingLanguageSimilarityModel(backend)
    elif method == "GIRAF":
        from .GIRAF.implementation import load_intrinsics
        context.intrinsics = load_intrinsics(prepared.intrinsics_path)
        context.depth_scale = options["depth_scale_meters"]
        context.depth_filter, context.geometry = options["depth_filter"], options["geometry"]
        context.semantic = {**options["semantic"], "model": config["vlm"]["model"]}
        if args.semantic_provider:
            context.semantic["provider"] = args.semantic_provider
    return context


def run_som(frame, instruction, perception, config, output):
    from .semantic.som import build_object_ids, parse_selection_count, run_semantic
    masks, names, scores, boxes = perception
    settings = config["segmentation"]
    options = config["methods"]["VLM-SoM"]
    object_ids = build_object_ids(names, settings.get("box_label_style", "number"))
    image = Image.open(frame.rgb_path).convert("RGB")
    annotated = overlay_results(image, boxes, masks, None, names,
                                settings.get("box_label_style", "number"))
    args = SimpleNamespace(
        model=config["vlm"]["model"], base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"), temperature=options.get("temperature"),
        reasoning_effort=options.get("reasoning_effort"), generations=options["generations"],
        selection_range=parse_selection_count(options["selections_per_generation"]),
        send_categories_to_vlm=settings.get("send_categories_to_vlm", False))
    started = time.perf_counter()
    probability, traces = run_semantic(annotated, {"instruction": instruction}, object_ids, names, args)
    point = frame.ground_truth_point
    matches = []
    if point and 0 <= point.y < masks.shape[1] and 0 <= point.x < masks.shape[2]:
        matches = np.flatnonzero(masks[:, point.y, point.x]).tolist()
    selected = max(matches, key=lambda i: float(scores[i])) if matches else None
    truth = ({"point": {"x": point.x, "y": point.y},
              "object_id": object_ids[selected] if selected is not None else None,
              "matching_object_ids": [object_ids[i] for i in matches]} if point else None)
    result = {"frame_id": frame.frame_id, "mode": "semantic", "instruction": instruction,
              "objects": [{"object_id": oid, "name": name, "semantic_probability": float(p)}
                          for oid, name, p in zip(object_ids, names, probability)],
              "ground_truth": truth, "semantic_traces": traces,
              "processing_time_seconds": time.perf_counter() - started}
    if config.get("visualize"):
        annotated.save(output / "som_prompt.jpg")
    return result


def infer_frame(method, frame, instruction, config, context, cache, output, grounder=None):
    output.mkdir(parents=True, exist_ok=True)
    if method == "geometry":
        from .geometry.implementation import process, load_intrinsics
        result = process(frame, load_intrinsics(context.prepared.intrinsics_path),
                         config["methods"][method], context.segmentation, output,
                         cache / "perception")
    elif method == "POMDP":
        from .POMDP.implementation import process_frame
        result = process_frame(frame, instruction, context, output)
    elif method == "GIRAF":
        from .GIRAF.implementation import process_frame
        result = process_frame(frame, {"instruction": instruction}, context, output)
    else:
        perception = load_perception(cache / "perception" / f"{frame.frame_id}.npz")
        masks, names, scores, boxes = perception
        if method == "VLM-SoM":
            result = run_som(frame, instruction, perception, config, output)
        else:
            result = grounder.ground(frame.rgb_path, instruction, None,
                                     "grid" if method == "VLM-Grid" else "pivot",
                                     output, batch_objects=(masks, names, scores))
            point = frame.ground_truth_point
            from .POMDP.implementation import ground_truth_index
            gt = ground_truth_index(point, masks, scores)
            distribution = result["object_distribution"]
            predicted = distribution[0]["object_index"] if distribution and distribution[0]["probability"] > 0 else None
            result.update(frame_id=frame.frame_id, ground_truth_index=gt,
                          predicted_index=predicted, correct=gt is not None and predicted == gt)
        np.savez_compressed(output / "perception.npz", masks=masks, box_names=np.asarray(names),
                            scores=scores, bboxes=boxes, image_size=np.asarray(Image.open(frame.rgb_path).size))
    result["baseline_method"] = method
    write_json(output / "result.json", result)
    return result


def main(default_method=None):
    args = build_parser(default_method).parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.visualize is None:
        args.visualize = bool(config.get("visualize", False))
    config["visualize"] = args.visualize
    if args.vocabulary:
        config["segmentation"]["vocabulary"] = args.vocabulary
    source = (args.input_dir or PROJECT_ROOT / config["input_dir"]).resolve()
    output_root = (args.output_dir or PROJECT_ROOT / config["output_dir"] / args.method).resolve()
    run = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cache_root = (args.prepared_dir or PROJECT_ROOT / "batch_preparation" / source.name / "input").resolve()
    analysis_input = cache_root if args.batch else run / "_prepared_input"
    if args.image:
        if args.batch or args.method not in {"VLM-Grid", "VLM-PIVOT", "VLM-SoM"}:
            raise SystemExit("A single RGB image supports live VLM methods only; use --input-dir for RGB-D methods")
        if not args.instruction:
            raise SystemExit("A single RGB image requires --instruction")
        group = args.group or "single"
        segmentation = segmentation_config_for_group(config, group)
        from migu.datasets import DatasetFrame
        image = args.image.resolve()
        frame = DatasetFrame(image.stem, image, None, None)
        single_cache = analysis_input / group
        path = single_cache / "perception" / f"{frame.frame_id}.npz"
        from batch_preparation.shared import save_perception
        save_perception(image, path, segmentation)
        work = [(group, None, [frame], single_cache, segmentation)]
    else:
        work = []
        for group, path in select_groups(source, args.subdirs if args.subdirs is not None else config.get("subdirs")):
            cache = cache_root / group if args.batch else analysis_input / group
            if args.batch:
                if not (cache / "perception").is_dir():
                    raise SystemExit(f"Missing prepared masks: {cache}; run python -m batch_preparation first")
                prepared = open_dataset(cache).prepare(cache)
            else:
                prepared, manifest = prepare_group(path, cache, config, group, overwrite=True,
                                                   visualize=args.visualize)
                if manifest["failures"]:
                    raise SystemExit(f"Preparation failed for {group}; inspect {cache / 'manifest.json'}")
            frames = pair_prepared_frames(prepared, require_gesture=args.method in {"geometry","POMDP","GIRAF"},
                                          require_valid_pointing_ray=args.method in {"geometry","POMDP","GIRAF"})
            work.append((group, prepared, frames, cache, segmentation_config_for_group(config, group)))
    grounder = None
    if args.method in {"VLM-Grid", "VLM-PIVOT"}:
        from .semantic.grounder import Grounder, GroundingConfig
        options = config["methods"][args.method]
        accepted = {f.name for f in fields(GroundingConfig)}
        grounder = Grounder(GroundingConfig(**{k:v for k,v in options.items() if k in accepted}),
                            model=config["vlm"]["model"])
    summaries = []
    failed = False
    for group, prepared, frames, cache, segmentation in work:
        context = method_context(args.method, config, prepared, segmentation, args, cache)
        context.prepared = prepared
        results, failures = [], []
        for frame in frames:
            instruction = args.instruction or (prepared.instruction_by_frame.get(frame.frame_id) if prepared else None)
            if args.method != "geometry" and not instruction:
                failures.append({"frame_id": frame.frame_id, "error": "Missing dataset instruction"})
                continue
            try:
                result = infer_frame(args.method, frame, instruction, config, context, cache,
                                     run / group / frame.frame_id, grounder)
                results.append(result)
                print(f"[{args.method}/{group}/{frame.frame_id}] complete")
            except Exception as error:
                failures.append({"frame_id": frame.frame_id, "error": str(error)})
                print(f"[{args.method}/{group}/{frame.frame_id}] failed: {error}")
        summary = {"group": group, "method": args.method, "frames": results,
                   "failures": failures, "input_root": str(analysis_input)}
        write_json(run / group / "summary.json", summary)
        summaries.append(summary)
        failed |= bool(failures) or not frames
    write_json(run / "summary.json", {"method": args.method, "groups": summaries,
                                      "input_root": str(analysis_input)})
    print(f"Output: {run}")
    if args.analyze:
        from .metrics import evaluate_batch_outputs
        evaluate_batch_outputs(output_root, analysis_input, run / "analysis",
                               method=args.method, run_name=run.name,
                               localization_point="bbox_center")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

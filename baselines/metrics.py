"""Unified metrics for grounding batch outputs."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

DET = {"giraf", "touch_line_np"}
MAIN_PROBABILITY_KEYS = {
    "fusion": "fused_probability",
    "semantic": "semantic_probability",
    "gesture": "gesture_probability",
}

def _json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def _norm(v):
    v=[max(0.,float(x)) if math.isfinite(float(x)) else 0. for x in v]; s=sum(v)
    return [x/s for x in v] if s else ([1/len(v)]*len(v) if v else [])
def _center(b):
    x1,y1,x2,y2=map(float,b); return ((x1+x2)/2,(y1+y2)/2)
def _gt_ids(g):
    v=g.get("matching_object_ids") or ([g["object_id"]] if g.get("object_id") is not None else [])
    return {str(x) for x in v}

def calculate_target_object_top_ratio(output_dir: str|Path)->float:
    vals=[]
    for p in ([Path(output_dir)] if Path(output_dir).is_file() else Path(output_dir).rglob("result.json")):
        d=_json(p); g=d.get("ground_truth"); objs=d.get("objects") or []
        if isinstance(g,dict) and _gt_ids(g) and objs:
            top=max(objs,key=lambda x:float(x.get("fused_probability",-math.inf)))
            vals.append(str(top.get("object_id")) in _gt_ids(g))
    if not vals: raise ValueError(f"No evaluable result.json files found below {output_dir}")
    return sum(vals)/len(vals)

def _base(method,run,group,fid,ids,gt,probs=None,det=False):
    hit=[x in gt for x in ids]
    r={"method":method,"run":run.name,"group":group,"frame_id":fid,
       "accuracy":float(any(hit[:1])),"top2_accuracy":float(any(hit[:2])),
       "top3_accuracy":float(any(hit[:3])),"nll":None,"brier_score":None}
    if det: r["top2_accuracy"]=r["top3_accuracy"]=r["accuracy"]
    elif probs is not None:
        q=sum(p for x,p in zip(ids,probs) if x in gt)
        r["nll"]=-math.log(max(q,1e-15))
        r["brier_score"]=(q-1)**2+sum(p*p for x,p in zip(ids,probs) if x not in gt)
    return r

def _with_time(row, frame):
    value=frame.get("processing_time_seconds")
    row["processing_time_seconds"]=(float(value) if value is not None else None)
    return row

def _add_preparation_times(rows, input_root, warn):
    """Add cached hand/segmentation latency to each method's per-frame time."""
    manifests={}
    warned=set()
    for row in rows:
        group=str(row.get("group") or "")
        frame_id=str(row.get("frame_id") or "")
        if group not in manifests:
            path=input_root/group/"manifest.json"
            try: manifests[group]=_json(path).get("frame_timings",{})
            except (OSError,json.JSONDecodeError,AttributeError): manifests[group]={}
        timing=manifests[group].get(frame_id,{})
        pipeline=row.get("processing_time_seconds")
        hand=timing.get("hand_detection_seconds")
        segmentation=timing.get("segmentation_seconds")
        row["pipeline_processing_time_seconds"]=pipeline
        row["hand_detection_time_seconds"]=(float(hand) if hand is not None else None)
        row["segmentation_time_seconds"]=(float(segmentation) if segmentation is not None else None)
        row["preparation_time_seconds"]=(
            float(hand)+float(segmentation)
            if hand is not None and segmentation is not None else None
        )
        row["processing_time_seconds"]=(
            float(pipeline)+row["preparation_time_seconds"]
            if pipeline is not None and row["preparation_time_seconds"] is not None
            else (float(pipeline) if pipeline is not None else None)
        )
        if frame_id and (hand is None or segmentation is None) and (group,frame_id) not in warned:
            warn.append(
                f"{group}/{frame_id}: batch preparation timing is missing; "
                "processing time excludes hand detection and segmentation"
            )
            warned.add((group,frame_id))
    return rows

def _timing_only(method,run,group,frame):
    row={"method":method,"run":run.name,"group":group,"frame_id":frame.get("frame_id"),
         "accuracy":None,"top2_accuracy":None,"top3_accuracy":None,"nll":None,
         "brier_score":None,"distance_2d_px":None,"distance_3d_m":None}
    return _with_time(row,frame)

def _perception_center(p,index):
    try:
        with np.load(p,allow_pickle=False) as d:
            box=np.asarray(d["bboxes"][index],dtype=float)
            # Perception archives store normalized xyxy boxes in current runs.
            if np.nanmax(np.abs(box)) <= 2 and "image_size" in d:
                width,height=map(float,d["image_size"]); box=box*np.array([width,height,width,height])
            return _center(box)
    except (OSError,KeyError,IndexError,ValueError): return None
def _gt_point(ds,fid):
    p=ds/"ground_truth.json"
    if not p.is_file(): return None
    q=_json(p).get(str(fid)); return (float(q["x"]),float(q["y"])) if q else None
def _intrinsics(p):
    d={}
    for line in p.read_text().splitlines():
        if ":" in line:
            k,v=line.split(":",1); d[k.strip()]=float(v)
    if not all(k in d for k in ("fx","fy","cx","cy")): raise ValueError("invalid intrinsics")
    return d
def _xyz(q,depth,k,scale):
    x=int(np.clip(round(q[0]),0,depth.shape[1]-1)); y=int(np.clip(round(q[1]),0,depth.shape[0]-1)); z=depth[y,x]
    if not np.isfinite(z) or z<=0:
        crop=depth[max(0,y-5):y+6,max(0,x-5):x+6]; valid=crop[np.isfinite(crop)&(crop>0)]
        if not valid.size:return None
        z=float(np.median(valid))
    z*=scale; return ((q[0]-k["cx"])*z/k["fx"],(q[1]-k["cy"])*z/k["fy"],z)
def _dist(r,pred,gt,ds,fid,scale,warn,pred_xyz=None):
    r.update(pred_x=pred[0] if pred else None,pred_y=pred[1] if pred else None,
             gt_x=gt[0] if gt else None,gt_y=gt[1] if gt else None,
             distance_2d_px=math.dist(pred,gt) if pred and gt else None,distance_3d_m=None)
    if not pred or not gt:return
    try:
        dep=np.asarray(Image.open(ds/"key_depth"/f"{fid}.png"),float); k=_intrinsics(ds/"camera_intrinsics.txt")
        a,b=(tuple(map(float,pred_xyz)) if pred_xyz is not None else _xyz(pred,dep,k,scale)),_xyz(gt,dep,k,scale)
        if a and b:r["distance_3d_m"]=math.dist(a,b)
    except (OSError,ValueError) as e:warn.append(f"{ds.name}/{fid}: {e}")

def _giraf_metric_point(frame,perception,index,warn):
    """Use GIRAF's ray/depth point only when it belongs to the selected mask."""
    hit=frame.get("ray_environment_point")
    if not isinstance(hit,dict) or hit.get("pixel") is None or hit.get("point_3d") is None:
        return None,None
    try:
        pixel=tuple(map(float,hit["pixel"])); xyz=tuple(map(float,hit["point_3d"]))
        if len(pixel)!=2 or len(xyz)!=3 or not all(map(math.isfinite,pixel+xyz)):return None,None
        x,y=map(int,pixel)
        with np.load(perception,allow_pickle=False) as data:
            masks=np.asarray(data["masks"])
            if 0<=index<len(masks) and 0<=y<masks.shape[1] and 0<=x<masks.shape[2] and bool(masks[index,y,x]):
                return pixel,xyz
    except (OSError,KeyError,IndexError,TypeError,ValueError) as e:
        warn.append(f"GIRAF point {perception}: {e}")
    return None,None

def _main_peak(fd,fid,index,warn):
    try:
        with np.load(fd/f"{fid}_posterior.npz") as a,np.load(fd/"perception.npz") as b:
            prob=np.asarray(a["probability"],float); mask=np.asarray(b["masks"])[index].astype(bool)
            valid=mask&np.isfinite(prob)
            if not valid.any():
                box=np.asarray(b["bboxes"][index],float)
                if np.nanmax(np.abs(box))<=2:
                    h,w=prob.shape;box=box*np.array([w,h,w,h])
                return _center(box)
            y,x=np.unravel_index(np.argmax(np.where(valid,prob,-np.inf)),prob.shape); return float(x),float(y)
    except (OSError,KeyError,IndexError,ValueError) as e:
        warn.append(f"posterior {fd}: {e}"); return _perception_center(fd/"perception.npz",index)

def _main_probability_key(data):
    """Return the score field produced by the active main pipeline branch."""
    mode=str(data.get("mode") or "fusion").lower()
    key=MAIN_PROBABILITY_KEYS.get(mode)
    if key is None:
        raise ValueError(f"unsupported main result mode: {mode!r}")
    return key

def _main_rows(method,run,inp,scale,warn,localization_point="posterior_peak"):
    out=[]
    for p in sorted(run.glob("*/*/result.json")):
        d=_json(p); g=d.get("ground_truth"); objs=d.get("objects") or []
        if not isinstance(g,dict) or not g.get("point") or not _gt_ids(g) or not objs:
            out.append(_timing_only(method,run,p.parent.parent.name,d));continue
        probability_key=_main_probability_key(d)
        indexed_objects=list(enumerate(objs))
        ranked_pairs=sorted(indexed_objects,key=lambda pair:float(pair[1].get(probability_key,0)),reverse=True)
        ranked=[item for _,item in ranked_pairs]
        ids=[str(x.get("object_id")) for x in ranked]; probs=_norm([x.get(probability_key,0) for x in ranked])
        r=_with_time(_base(method,run,p.parent.parent.name,d["frame_id"],ids,_gt_ids(g),probs),d)
        gt=(float(g["point"]["x"]),float(g["point"]["y"]))
        top_index=ranked_pairs[0][0]
        pred=(_perception_center(p.parent/"perception.npz",top_index)
              if localization_point == "bbox_center" or d.get("mode") == "semantic"
              else _main_peak(p.parent,d["frame_id"],top_index,warn))
        _dist(r,pred,gt,inp/p.parent.parent.name,d["frame_id"],scale,warn); out.append(r)
    return out

def _summary_rows(method,run,inp,scale,warn):
    out=[]; low={"pomdp":"legs","vlm-grid":"semantic_grounding_grid","vlm-pivot":"semantic_grounding_pivot"}.get(method.lower(),method.lower())
    for s in sorted(run.glob("*/summary.json")):
        group=s.parent.name
        for f in _json(s).get("frames",[]):
            gt,fid=f.get("ground_truth_index"),f.get("frame_id")
            row_method=method
            if gt is None or not fid:
                out.append(_timing_only(row_method,run,group,f));continue
            if low=="giraf":
                target_index=f.get("target_index")
                ids=[] if target_index is None else [str(target_index)]
                probs=None
            elif low=="legs":
                z=f.get("ranked_candidates") or []; ids=[str(x["id"]) for x in z]; probs=_norm([x.get("fused_probability",0) for x in z])
            elif low.startswith("semantic_grounding"):
                z=f.get("top_objects") or f.get("object_distribution") or []; z=sorted(z,key=lambda x:float(x.get("probability",0)),reverse=True)
                ids=[str(x["object_index"]) for x in z]; probs=_norm([x.get("probability",0) for x in z])
            else:continue
            if not ids:
                if low=="giraf":
                    r=_with_time(_base(row_method,run,group,fid,[],{str(gt)},None,True),f)
                    r["distance_point_source"]=None
                    _dist(r,None,_gt_point(inp/group,fid),inp/group,fid,scale,warn)
                    out.append(r)
                continue
            r=_with_time(_base(row_method,run,group,fid,ids,{str(gt)},probs,low in DET),f)
            perception=inp/group/"perception"/f"{fid}.npz"; index=int(ids[0])
            pred_xyz=None
            if low=="giraf":
                pred,pred_xyz=_giraf_metric_point(f,perception,index,warn)
                r["distance_point_source"]="ray_environment_point" if pred is not None else "object_box_center"
            else: pred=None
            if pred is None:pred=_perception_center(perception,index)
            _dist(r,pred,_gt_point(inp/group,fid),inp/group,fid,scale,warn,pred_xyz); out.append(r)
    return out

def _touch_rows(method,run,inp,scale,warn):
    out=[]
    for s in sorted(run.glob("*/summary.json")):
        group=s.parent.name
        for f in _json(s).get("frames",[]):
            fid,box=f.get("frame_id"),f.get("box_xyxy"); gt=_gt_point(inp/group,fid) if fid else None
            if not fid or box is None or gt is None:
                out.append(_timing_only(method,run,group,f));continue
            ok=bool(f.get("ground_truth_point_inside",box[0]<=gt[0]<=box[2] and box[1]<=gt[1]<=box[3]))
            r={"method":method,"run":run.name,"group":group,"frame_id":fid,"accuracy":float(ok),"top2_accuracy":float(ok),"top3_accuracy":float(ok),"nll":None,"brier_score":None}
            _with_time(r,f)
            _dist(r,_center(box),gt,inp/group,fid,scale,warn); out.append(r)
    return out

def _geometric_rows(method,run,inp,scale,warn):
    out=[]
    for p in sorted(run.glob("*/*/result.json")):
        f=_json(p); gt=f.get("ground_truth") or {}; index=gt.get("candidate_index")
        fid=f.get("frame_id")
        if index is None or not fid:
            out.append(_timing_only(method,run,p.parent.parent.name,f));continue
        ranked=np.argsort(-np.asarray(f.get("probability",[]),dtype=float)).tolist()
        if not ranked:continue
        r=_with_time(_base(method,run,p.parent.parent.name,fid,[str(x) for x in ranked],{str(index)},_norm([f["probability"][x] for x in ranked])),f)
        perception=inp/p.parent.parent.name/"perception"/f"{fid}.npz"
        pred=_perception_center(perception,ranked[0])
        point=gt.get("point"); target=(float(point["x"]),float(point["y"])) if point else None
        _dist(r,pred,target,inp/p.parent.parent.name,fid,scale,warn);out.append(r)
    return out

def _aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r["method"],r["run"])].append(r)
    out=[]; metrics=("accuracy","top2_accuracy","top3_accuracy","distance_2d_px","distance_3d_m","nll","brier_score")
    for (m,run),items in sorted(groups.items()):
        x={"method":m,"run":run,"frames":len(items)}
        for key in metrics:
            v=[float(r[key]) for r in items if r.get(key) is not None]; x[key]=sum(v)/len(v) if v else None; x[key+"_count"]=len(v)
        distances_2d=[float(r["distance_2d_px"]) for r in items if r.get("distance_2d_px") is not None]
        distances_3d=[float(r["distance_3d_m"]) for r in items if r.get("distance_3d_m") is not None]
        x["distance_2d_variance_px2"]=float(np.var(distances_2d)) if distances_2d else None
        x["distance_3d_variance_m2"]=float(np.var(distances_3d)) if distances_3d else None
        for task in ("pick", "place"):
            task_items=[r for r in items if task in str(r.get("group", "")).lower()]
            task_2d=[float(r["distance_2d_px"]) for r in task_items if r.get("distance_2d_px") is not None]
            task_3d=[float(r["distance_3d_m"]) for r in task_items if r.get("distance_3d_m") is not None]
            x[f"{task}_distance_2d_px"]=float(np.mean(task_2d)) if task_2d else None
            x[f"{task}_distance_2d_variance_px2"]=float(np.var(task_2d)) if task_2d else None
            x[f"{task}_distance_2d_px_count"]=len(task_2d)
            x[f"{task}_distance_3d_m"]=float(np.mean(task_3d)) if task_3d else None
            x[f"{task}_distance_3d_variance_m2"]=float(np.var(task_3d)) if task_3d else None
            x[f"{task}_distance_3d_m_count"]=len(task_3d)
        times=[float(r["processing_time_seconds"]) for r in items if r.get("processing_time_seconds") is not None]
        x["processing_time_mean_seconds"]=float(np.mean(times)) if times else None
        x["processing_time_variance_seconds2"]=float(np.var(times)) if times else None
        x["processing_time_count"]=len(times)
        for key in ("pipeline_processing_time_seconds","hand_detection_time_seconds",
                    "segmentation_time_seconds","preparation_time_seconds"):
            values=[float(r[key]) for r in items if r.get(key) is not None]
            x[key.replace("_seconds","_mean_seconds")]=float(np.mean(values)) if values else None
        out.append(x)
    return out
def _csv(p,rows):
    if not rows:p.write_text("");return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def _plot(rows,p,warn):
    if not rows:return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:warn.append("matplotlib unavailable; no plot created");return
    ms=[("accuracy","Accuracy"),("top2_accuracy","Top-2 accuracy"),("top3_accuracy","Top-3 accuracy"),("distance_2d_px","2-D distance (px)"),("distance_3d_m","3-D distance (m)"),("nll","NLL"),("brier_score","Brier score")]; labels=[x["method"]+"\n"+x["run"] for x in rows]
    fig,ax=plt.subplots(3,3,figsize=(max(14,len(rows)*1.5),12))
    for a,(k,t) in zip(ax.flat,ms):
        a.bar(range(len(rows)),[np.nan if x[k] is None else x[k] for x in rows]);a.set_title(t)
        a.set_xticks(range(len(rows)));a.set_xticklabels(labels,rotation=55,ha="right",fontsize=7);a.grid(axis="y",alpha=.25)
    for a in ax.flat[len(ms):]:a.remove()
    fig.tight_layout();fig.savefig(p,dpi=180,bbox_inches="tight");plt.close(fig)

def _plot_time(rows,p,warn):
    timed=[x for x in rows if x.get("processing_time_mean_seconds") is not None]
    if not timed:return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:return
    labels=[x["method"]+"\n"+x["run"] for x in timed]; positions=np.arange(len(timed)); width=.38
    fig,left=plt.subplots(figsize=(max(9,len(timed)*1.8),6));right=left.twinx()
    mean=left.bar(positions-width/2,[x["processing_time_mean_seconds"] for x in timed],width,label="Mean time",color="#4C78A8")
    variance=right.bar(positions+width/2,[x["processing_time_variance_seconds2"] for x in timed],width,label="Variance",color="#F58518")
    left.set_ylabel("Mean processing time (s)");right.set_ylabel("Processing-time variance (s²)")
    left.set_xticks(positions);left.set_xticklabels(labels,rotation=35,ha="right");left.grid(axis="y",alpha=.25)
    left.set_title("Per-frame processing time by method");left.legend([mean,variance],["Mean time (s)","Variance (s²)"],loc="upper left")
    fig.tight_layout();fig.savefig(p,dpi=180,bbox_inches="tight");plt.close(fig)

def evaluate_batch_outputs(output_root:str|Path,input_root:str|Path|None=None,save_dir:str|Path|None=None,depth_scale:float=.001,method:str|None=None,localization_point:str="posterior_peak",run_name:str|None=None)->dict[str,Any]:
    if localization_point not in {"bbox_center", "posterior_peak"}:
        raise ValueError("localization_point must be 'bbox_center' or 'posterior_peak'")
    root=Path(output_root).resolve(); inp=Path(input_root).resolve() if input_root else root.parent/"input"; dest=Path(save_dir).resolve() if save_dir else Path(__file__).parent/"metric_results";dest.mkdir(parents=True,exist_ok=True)
    rows=[];warn=[]
    method_dirs=[(x.name,x) for x in sorted(root.iterdir()) if x.is_dir()]
    if method is not None:
        method_dirs=[(name,path) for name,path in method_dirs if name.casefold()==method.casefold()]
        # Some standalone baselines use OUTPUTS/RUN/GROUP/summary.json rather
        # than OUTPUT_ROOT/METHOD/RUN/GROUP/summary.json.  With an explicit
        # method, accept that native outputs directory directly.
        if not method_dirs and any(root.glob("*/*/summary.json")):
            method_dirs=[(method,root)]
        if not method_dirs:
            available=", ".join(x.name for x in sorted(root.iterdir()) if x.is_dir()) or "none"
            raise ValueError(f"Method {method!r} not found below {root}; available methods: {available}")
    for method_name,md in method_dirs:
        runs = [x for x in md.iterdir() if x.is_dir() and (run_name is None or x.name == run_name)]
        if not runs:
            continue
        # Run directories use sortable timestamps (YYYYMMDD_HHMMSS_microseconds).
        # Evaluate only the newest result from each method.
        run = max(runs, key=lambda x: x.name)
        low=method_name.lower()
        rows += (_main_rows(method_name,run,inp,depth_scale,warn,localization_point) if low in {"main", "vlm-som", "semantic_som"} else
                 _touch_rows(method_name,run,inp,depth_scale,warn) if low=="touch_line_np" else
                 _geometric_rows(method_name,run,inp,depth_scale,warn) if low in {"geometric_based", "geometry"} else
                 _summary_rows(method_name,run,inp,depth_scale,warn))
    _add_preparation_times(rows,inp,warn)
    summary=_aggregate(rows)
    for item in summary:
        if item["processing_time_count"]==0:warn.append(f"{item['method']}/{item['run']}: processing_time_seconds is missing (legacy output)")
    _csv(dest/"metrics_per_frame.csv",rows);_csv(dest/"metrics_summary.csv",summary);_plot(summary,dest/"metrics_overview.png",warn);_plot_time(summary,dest/"processing_time.png",warn)
    report={"output_root":str(root),"input_root":str(inp),"depth_scale":depth_scale,"localization_point":localization_point,"runs":summary,"warnings":warn};(dest/"metrics_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False));return report

def evaluate_main_outputs(output_root:str|Path="batch_preparation/captures/output",input_root:str|Path|None=None,save_dir:str|Path|None=None,depth_scale:float=.001,localization_point:str="posterior_peak")->dict[str,Any]:
    """Evaluate the newest batch-preparation ``main`` run with every metric.

    ``output_root`` is the batch output directory containing the ``main``
    method directory.  Results use the same CSV, JSON and plot format as the
    unified evaluator, while restricting discovery to ``main``.
    """
    return evaluate_batch_outputs(
        output_root, input_root, save_dir, depth_scale, method="main",
        localization_point=localization_point,
    )

def evaluate_touch_line_np_outputs(output_root:str|Path="batch_preparation/captures/output",input_root:str|Path|None=None,save_dir:str|Path|None=None,depth_scale:float=.001)->dict[str,Any]:
    """Evaluate the newest ``touch_line_np`` run with all applicable metrics.

    ``touch_line_np`` emits one selected box rather than a ranked probability
    distribution.  Consequently top-2/top-3 are identical to top-1, while NLL
    and Brier score are reported as ``None``.  Localisation, task-specific
    localisation, and processing-time metrics use the unified evaluator.
    """
    return evaluate_batch_outputs(
        output_root, input_root, save_dir, depth_scale, method="touch_line_np"
    )

# A concise name for callers that import this module specifically for main.
evaluation = evaluate_main_outputs


def main():
    """Analyze the newest run of each public baseline, or one selected method."""
    import argparse
    from batch_preparation.shared import PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", type=Path,
                        default=PROJECT_ROOT / "output" / "baselines")
    parser.add_argument("--input-root", type=Path,
                        default=PROJECT_ROOT / "batch_preparation" / "captures" / "input")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--run", help="Select an exact timestamp instead of the newest run")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    args = parser.parse_args()
    report = evaluate_batch_outputs(
        args.output_root, args.input_root,
        args.save_dir or args.output_root / "metrics_results", args.depth_scale,
        method=args.method, run_name=args.run, localization_point="bbox_center")
    if not report["runs"]:
        raise SystemExit("No evaluable baseline results were found")
    for row in report["runs"]:
        print(f"{row['method']}/{row['run']}: accuracy={row['accuracy']}, frames={row['frames']}")


if __name__ == "__main__":
    main()

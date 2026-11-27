"""Score LArTPC: python scripts/panda/eval/test_lartpc.py --config CONFIG --checkpoint WEIGHTS --out DIR."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from ._common import (jsonable, load_config, load_model, loader, make_dataset, numpy,
                          preserve_hits, save_json, seed, to_device)
else:
    from _common import (jsonable, load_config, load_model, loader, make_dataset, numpy,
                         preserve_hits, save_json, seed, to_device)

MOTIF5 = ["shower", "track", "michel", "delta", "led"]
JOINT8 = ["shower+photon", "shower+electron", "track+muon", "track+pion", "track+proton",
          "michel+electron", "delta+electron", "led+none"]
MOTIF = [0, 0, 1, 1, 1, 2, 3, 4]
PID = ["other", "pion", "kaon", "proton", "electron"]


def prf(cm, all_classes=False):
    cm = np.asarray(cm, dtype=np.float64)
    n = len(cm)
    tp, true, pred = np.diag(cm), cm.sum(1), cm.sum(0)[:n]
    precision = np.divide(tp, pred, out=np.zeros(n), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros(n), where=true > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=precision + recall > 0)
    seen = np.ones(n, dtype=bool) if all_classes else (true > 0) | (pred > 0)
    mp, mr = precision[seen].mean(), recall[seen].mean()
    result = dict(accuracy=tp.sum() / max(cm.sum(), 1), macro_precision=mp, macro_recall=mr,
                  macro_f1=f1[seen].mean(), precision=precision, recall=recall, f1=f1,
                  support=true, confusion=cm.astype(np.int64))
    if all_classes:
        result["f1_of_macro"] = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    return result


def semantic_cm(prediction, classes, noise):
    true = np.where(prediction["gt_noise"], noise, prediction["gt_class"])
    pred = np.where(prediction["pred_instance"] >= 0, prediction["pred_class"],
                    np.where(prediction["pred_noise"], noise, classes))
    pred = np.where((pred < 0) | (pred > classes), classes, pred)
    keep = (true >= 0) & (true < classes)
    return np.bincount(true[keep] * (classes + 1) + pred[keep],
                       minlength=classes * (classes + 1)).reshape(classes, classes + 1)


def decode(model, batch, output, label, detector):
    from pimm.engines.metrics.panoptic import noise_points, scored_points

    spec = model.label_specs[label]
    segment = spec["segment_key"] if spec["segment_key"] in batch else spec["segment_fallback_key"]
    truth, classes, voxel = scored_points(batch, spec["instance_key"], segment)
    if isinstance(voxel, slice):
        raise ValueError("Hit-level scoring requires inverse and origin truth in the evaluation transforms")
    results = model.postprocess(output, label=label, decoding="panda_v2", conf_threshold=0.3,
                                mask_threshold=0.5, min_points=2, fill_uncovered=False)
    instances = numpy(results["instance_labels"]).reshape(-1).astype(np.int64)
    # Filled hits inherit the kept query's noise call through its original owned voxels.
    noise_results = dict(results, instance_labels=results["instance_labels"].masked_fill(
        results["query_labels"] < 0, -1))
    gt_noise, pred_noise = noise_points(batch, spec, noise_results, truth, classes, voxel, instances)
    if detector == "lartpc" and spec.get("stuff_classes"):
        task = output["outputs_by_label"][label]
        pred_noise = numpy(task["stuff_probs"].float() >= 0.5).reshape(-1)[voxel]
    return dict(gt_instance=truth, gt_class=classes, pred_instance=instances[voxel],
                pred_class=numpy(results["class_labels"]).reshape(-1)[voxel].astype(np.int64),
                gt_noise=gt_noise, pred_noise=pred_noise)


@torch.no_grad()
def score_panseg(model, dataset, indices, out, detector="lartpc", device="cuda", workers=0):
    from pimm.engines.metrics.panoptic import event_row, noise_class, summarize_rows

    rows, cms, noise_cms = {}, {}, {}
    for number, (event, batch) in enumerate(zip(indices, loader(dataset, indices, workers=workers)), 1):
        batch = to_device(batch, device)
        output = model(batch, return_point=True)
        for label, spec in model.label_specs.items():
            prediction = decode(model, batch, output, label, detector)
            classes, noise = int(spec["num_classes"]), noise_class(spec)
            row = event_row(**prediction, num_classes=classes, noise_class=noise,
                            iou_threshold=0.5)
            row.update(event=event, hits=len(prediction["gt_instance"]))
            rows.setdefault(label, []).append(row)
            if detector == "lartpc":
                cm = semantic_cm(prediction, classes, noise)
            else:
                true = prediction["gt_class"]
                pred = np.where(prediction["pred_instance"] >= 0, prediction["pred_class"], classes)
                cm = np.bincount(true * (classes + 1) + pred,
                                 minlength=classes * (classes + 1)).reshape(classes, classes + 1)
                flag = numpy(batch["origin_segment_noise"]).reshape(-1).astype(np.int64)
                noise_cms.setdefault(label, []).append(np.bincount(
                    flag * 2 + prediction["pred_noise"].astype(np.int64), minlength=4).reshape(2, 2))
            cms.setdefault(label, []).append(cm)
        if number % 500 == 0 or number == len(indices):
            print(f"{number}/{len(indices)} images", flush=True)
    summary = dict(events=len(indices), labels={})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for label, event_rows in rows.items():
        spec = model.label_specs[label]
        record = summarize_rows(event_rows, int(spec["num_classes"]), noise_class(spec))
        record["semantic"] = prf(np.sum(cms[label], axis=0), all_classes=detector == "sphenix")
        arrays = {key: np.asarray([row[key] for row in event_rows]) for key in event_rows[0]}
        arrays["semantic_cm"] = np.asarray(cms[label])
        if label in noise_cms:
            arrays["noise_cm"] = np.asarray(noise_cms[label])
            record["noise"] = prf(arrays["noise_cm"].sum(0), all_classes=True)
        np.savez_compressed(out / f"events_{label}.npz", **arrays)
        summary["labels"][label] = record
    return summary


@torch.no_grad()
def score_semseg(model, heads, names, dataset, indices, batch_size, out, detector="lartpc",
                 device="cuda", workers=0):
    records, counts = {}, []
    for batch in loader(dataset, indices, batch_size=batch_size, workers=workers):
        batch = to_device(batch, device)
        output = model(batch)  # logits and labels at every hit
        hits = output["offset"].diff(prepend=output["offset"].new_zeros(1))
        events = len(hits)
        hit_event = torch.arange(events, device=device).repeat_interleave(hits)
        counts.extend(numpy(hits).tolist())
        for head, target in heads.items():
            classes = list(names[target])
            sets = {"motif5" if classes == MOTIF5 else "joint8" if classes == JOINT8 else target:
                    (output[target], None, classes)}
            if detector == "lartpc" and classes == JOINT8:
                sets["motif5"] = (batch["origin_segment_motif"], torch.tensor(MOTIF, device=device), MOTIF5)
            pred = output["seg_logits"][head].float().argmax(1)
            for name, (labels, merge, class_names) in sets.items():
                true = labels.view(-1).long()
                k = len(class_names)
                keep = (true >= 0) & (true < k)
                if detector == "sphenix" and not bool(keep.all()):
                    raise ValueError(f"{head}: every hit must have a valid label")
                prediction = pred[keep] if merge is None else merge[pred[keep]]
                cm = torch.bincount((hit_event[keep] * k + true[keep]) * k + prediction,
                                     minlength=events * k * k).view(events, k, k)
                records.setdefault((head, name), []).append(numpy(cm).astype(np.int32))
    summary = dict(events=len(counts), hits=int(sum(counts)), heads={})
    arrays = dict(event=np.asarray(indices), hits=np.asarray(counts))
    for (head, name), parts in records.items():
        matrices = np.concatenate(parts)
        arrays[f"{head}_{name}_cm"] = matrices
        summary["heads"].setdefault(head, {})[name] = prf(matrices.sum(0, dtype=np.int64),
                                                        all_classes=detector == "sphenix")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "events.npz", **arrays)
    return summary


def main(detector="lartpc"):
    name = "LArTPC" if detector == "lartpc" else "sPHENIX"
    parser = argparse.ArgumentParser(description=f"Score {name} checkpoints on raw test hits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head-checkpoint", action="append", default=[], help="Partial best-head checkpoint; repeat for each frozen probe group")
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--task", choices=("auto", "semantic", "instance"), default="auto")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="Semantic inference batch size")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.start < 0 or (args.max_events is not None and args.max_events <= 0):
        parser.error("start must be nonnegative and max-events must be positive")
    seed(args.seed)
    cfg = load_config(args.config)
    hook = next((h for h in cfg.get("hooks", []) if h["type"] == "PointClassificationEvaluator"), None)
    task = ("semantic" if hook else "instance") if args.task == "auto" else args.task
    if task == "semantic" and hook is None:
        parser.error("Semantic config needs PointClassificationEvaluator head and class-name mappings")
    cfg = copy.deepcopy(cfg)
    if detector == "sphenix":
        cfg.data[args.split].update(min_points=0, max_points=3200)
    if task == "semantic" and detector == "lartpc" and list(hook["names"]["segment"]) == JOINT8:
        preserve_hits(cfg.data[args.split], segment_motif="origin_segment_motif")
    limit = None if args.max_events is None else args.start + args.max_events
    dataset = make_dataset(cfg, args.split, args.data_root, max_events=limit)
    indices = list(range(args.start, len(dataset)))
    if not indices:
        parser.error("No images selected")
    model = load_model(cfg, args.checkpoint, args.device, overlays=args.head_checkpoint,
                       disable_flash=task == "instance")
    if task == "instance":
        summary = score_panseg(model, dataset, indices, args.out, detector, args.device, args.workers)
    else:
        summary = score_semseg(model, hook["heads"], hook["names"], dataset, indices, args.batch_size,
                               args.out, detector, args.device, args.workers)
    save_json(Path(args.out) / "summary.json", summary)
    print(json.dumps(jsonable(summary)), flush=True)


if __name__ == "__main__":
    main()

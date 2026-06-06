"""
Tester for semantic and panoptic segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import csv
import json
import numpy as np
from collections import OrderedDict
from datetime import datetime, timezone
import torch
import torch.distributed as dist
import pointops
from sklearn.metrics import adjusted_rand_score

import pimm.utils.comm as comm
from pimm.engines.metrics import (
    aggregate_instance_results,
    compute_semseg_metrics,
    eval_instances,
)
from pimm.distributed import create_parallel_context, prepare_model, unwrap_model
from pimm.datasets import build_dataset, collate_fn
from pimm.models import build_model
from pimm.utils.logger import get_root_logger
from pimm.utils.registry import Registry
from pimm.utils.misc import (
    intersection_and_union_gpu,
)
from pimm.models.utils.misc import offset2bincount


TESTERS = Registry("testers")


def _jsonable(value):
    """Convert tensors and numpy values to JSON-compatible containers."""
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path, payload):
    """Write newline-terminated JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, default=str)
        f.write("\n")


def _write_csv(path, rows):
    """Write rows with the union of all keys as CSV columns."""
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.parallel_context = create_parallel_context(cfg)
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def eval_output_dir(self):
        """Return the per-run eval artifact directory for this test process."""
        explicit_path = getattr(self.cfg, "eval_save_path", None)
        if explicit_path:
            path = explicit_path
        else:
            timestamp = getattr(self.cfg, "_eval_timestamp", None)
            if timestamp is None:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                object.__setattr__(self.cfg, "_eval_timestamp", timestamp)
            path = os.path.join(self.cfg.save_path, "eval", timestamp)
        os.makedirs(path, exist_ok=True)
        return path

    def write_eval_artifacts(self, task, metrics, rows=None):
        """Write structured test metrics from rank zero."""
        if not comm.is_main_process():
            return
        output_dir = self.eval_output_dir()
        payload = {
            "task": task,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "save_path": self.cfg.save_path,
            "weight": getattr(self.cfg, "weight", None),
            "config_file": getattr(self.cfg, "_config_file", None),
            "metrics": metrics,
        }
        json_path = os.path.join(output_dir, "test_metrics.json")
        csv_path = os.path.join(output_dir, "test_metrics.csv")
        _write_json(json_path, payload)
        if rows:
            _write_csv(csv_path, rows)
        self.logger.info(f"Wrote eval artifacts: {output_dir}")

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = prepare_model(model, self.cfg, self.parallel_context)
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, map_location="cpu", weights_only=False)
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                model_state = checkpoint.get("model", None)
                if isinstance(model_state, dict) and "state_dict" in model_state:
                    state_dict = model_state["state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif model_state is not None:
                    state_dict = model_state
            weight = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1 and hasattr(model, "module"):
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError


@TESTERS.register_module()
class SemSegTester(TesterBase):
    """Semantic segmentation tester following SemSegEvaluator pattern."""

    def __init__(
        self,
        cfg,
        model=None,
        test_loader=None,
        verbose=False,
        ignore_index=-1,
        macro_ignore_class_ids=None,
    ):
        super().__init__(cfg, model, test_loader, verbose)
        self.ignore_index = ignore_index
        self.macro_ignore_class_ids = tuple(sorted(set(macro_ignore_class_ids or [])))

    def test(self):
        self.logger.info(">>>>>>>>>>>>>>>> Start Semantic Segmentation Test >>>>>>>>>>>>>>>>")
        self.model.eval()

        all_preds = []
        all_segments = []

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.model(input_dict)

            if "seg_logits" in output_dict:
                output = output_dict["seg_logits"]
            elif "sem_logits" in output_dict:
                output = output_dict["sem_logits"]
            else:
                raise KeyError(
                    "No semantic logits found in model output (expected 'seg_logits' or 'sem_logits')."
                )

            pred = output.max(1)[1]
            segment = input_dict["segment"]

            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]

            segment = segment.squeeze(-1)

            all_preds.append(pred.cpu())
            all_segments.append(segment.cpu())

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.cfg.data.num_classes,
                self.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)

            info = "Test: [{iter}/{max_iter}]".format(
                iter=i + 1, max_iter=len(self.test_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.logger.info(info)

        if comm.get_world_size() > 1:
            all_preds_gathered = comm.gather(all_preds, dst=0)
            all_segments_gathered = comm.gather(all_segments, dst=0)
            if comm.get_rank() == 0:
                all_preds = [p for preds in all_preds_gathered for p in preds]
                all_segments = [s for segments in all_segments_gathered for s in segments]
            else:
                self.model.train()
                return

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_segments = torch.cat(all_segments, dim=0).numpy()

        num_classes = self.cfg.data.num_classes
        metrics = compute_semseg_metrics(
            all_preds,
            all_segments,
            num_classes,
            macro_ignore_class_ids=self.macro_ignore_class_ids,
        )
        precision_class = metrics.precision_class
        recall_class = metrics.recall_class
        f1_class = metrics.f1_class
        iou_class = metrics.iou_class
        acc_class = metrics.acc_class
        macro_mask = metrics.macro_mask
        m_precision = metrics.m_precision
        m_recall = metrics.m_recall
        m_f1 = metrics.m_f1
        m_iou = metrics.m_iou
        m_acc = metrics.m_acc
        all_acc = metrics.all_acc

        self.logger.info(
            "Test result: mIoU/mAcc/allAcc/mPrec/mRec/mF1 {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc, m_precision, m_recall, m_f1
            )
        )

        table_header = "| Class ID | Class Name | IoU | Accuracy | Precision | Recall | F1 |"
        table_separator = (
            "|"
            + "-" * 10
            + "|"
            + "-" * 12
            + "|"
            + "-" * 8
            + "|"
            + "-" * 10
            + "|"
            + "-" * 11
            + "|"
            + "-" * 8
            + "|"
            + "-" * 6
            + "|"
        )

        self.logger.info("Per-class metrics:")
        self.logger.info(table_header)
        self.logger.info(table_separator)

        if not macro_mask.all():
            self.logger.info("* indicates class ignored in macro metrics")

        for i in range(self.cfg.data.num_classes):
            ignored_marker = "*" if not macro_mask[i] else ""
            self.logger.info(
                "| {idx:8d} | {name:10s} | {iou:.4f} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                    idx=i,
                    name=(self.cfg.data.names[i] + ignored_marker),
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                    precision=precision_class[i],
                    recall=recall_class[i],
                    f1=f1_class[i],
                )
            )

        class_names = list(getattr(self.cfg.data, "names", []))
        per_class = []
        for i in range(num_classes):
            name = class_names[i] if i < len(class_names) else f"class_{i}"
            per_class.append(
                {
                    "class_id": i,
                    "class_name": name,
                    "macro_included": bool(macro_mask[i]),
                    "iou": float(iou_class[i]),
                    "accuracy": float(acc_class[i]),
                    "precision": float(precision_class[i]),
                    "recall": float(recall_class[i]),
                    "f1": float(f1_class[i]),
                }
            )
        summary = {
            "m_iou": float(m_iou),
            "m_acc": float(m_acc),
            "all_acc": float(all_acc),
            "m_precision": float(m_precision),
            "m_recall": float(m_recall),
            "m_f1": float(m_f1),
        }
        self.write_eval_artifacts(
            "semseg",
            {"summary": summary, "per_class": per_class},
            [{"row_type": "summary", **summary}]
            + [{"row_type": "per_class", **row} for row in per_class],
        )

        self.logger.info("<<<<<<<<<<<<<<<<< End Semantic Segmentation Test <<<<<<<<<<<<<<<<<")
        self.model.train()


@TESTERS.register_module()
class InstanceSegTester(TesterBase):
    """Panoptic/Instance segmentation tester following InstanceSegmentationEvaluator pattern."""

    def __init__(
        self,
        cfg,
        model=None,
        test_loader=None,
        verbose=False,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        class_names=None,
        stuff_classes=None,
        iou_thresh=0.5,
        require_class_for_match=False,
    ):
        super().__init__(cfg, model, test_loader, verbose)
        self.stuff_threshold = float(stuff_threshold)
        self.mask_threshold = float(mask_threshold)
        self.iou_thresh = float(iou_thresh)
        self.require_class_for_match = bool(require_class_for_match)
        self.class_names = tuple(class_names or [])
        self.stuff_classes = tuple(sorted(stuff_classes or ()))

    def test(self):
        self.logger.info(
            ">>>>>>>>>>>>>> Start Panoptic/Instance Segmentation Test >>>>>>>>>>>>>>>>"
        )
        self.model.eval()

        class_names = (
            self.class_names
            if len(self.class_names)
            else tuple(getattr(self.cfg.data, "names", []))
        )
        if not class_names:
            class_names = tuple(range(self.cfg.data.num_classes))

        all_stats = []
        ari_scores = []
        momentum_stats = []

        for input_dict in self.test_loader:
            assert (
                len(input_dict["offset"]) == 1
            ), "InstanceSegTester requires bs=1"

            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.model(input_dict, return_point=True)

            point = output_dict.get("point")
            if point is None:
                self.logger.warning("InstanceSegTester: missing point data")
                continue

            gt_instance = point.instance
            if gt_instance is None:
                self.logger.warning("InstanceSegTester: missing instance labels")
                continue

            gt_segment = getattr(point, "segment", None)
            if gt_segment is None:
                self.logger.warning("InstanceSegTester: missing PID labels")
                continue

            pred_masks_list = output_dict.get("pred_masks")
            pred_logits_list = output_dict.get("pred_logits")
            pred_momentum_list = output_dict.get("pred_momentum")
            if not pred_masks_list or pred_logits_list is None:
                self.logger.warning("InstanceSegTester: missing predictions")
                continue

            model = unwrap_model(self.model)

            stuff_probs = (
                point.outputs.get("stuff_probs")
                if hasattr(point, "outputs")
                else None
            )
            point_counts = offset2bincount(point.offset)

            post_input = {
                "pred_masks": pred_masks_list,
                "pred_logits": pred_logits_list,
                "stuff_probs": stuff_probs,
                "point_counts": point_counts,
                "pred_momentum": pred_momentum_list,
            }

            results = model.postprocess(
                post_input,
                stuff_threshold=self.stuff_threshold,
                mask_threshold=self.mask_threshold,
            )

            pred_instance_labels = results["instance_labels"].cpu()
            pred_pid_labels = results["class_labels"].cpu()

            pred_instance_momentum = None
            if "pred_momentum" in results:
                pred_instance_momentum = results["pred_momentum"].cpu()

            gt_inst = gt_instance.squeeze(-1).cpu().numpy().astype(np.int64)
            gt_pid = gt_segment.squeeze(-1).cpu().numpy().astype(np.int64)
            pr_inst = pred_instance_labels.numpy().astype(np.int64)
            pr_pid = pred_pid_labels.numpy().astype(np.int64)

            stats = eval_instances(
                gt_inst,
                pr_inst,
                gt_pid,
                pr_pid,
                class_names=class_names,
                iou_thresh=self.iou_thresh,
                require_class_for_match=self.require_class_for_match,
            )
            all_stats.append(stats)

            momentum_gt = input_dict.get("momentum")
            if pred_instance_momentum is not None and momentum_gt is not None:
                counts = offset2bincount(point.offset)
                counts_list = counts.cpu().tolist()

                criterion = model.criteria.criteria[0].criterion

                mom_stats = self._eval_momentum(
                    stats["matches"],
                    pred_instance_momentum,
                    momentum_gt,
                    gt_instance,
                    counts_list,
                    criterion,
                    num_classes=self.cfg.data.num_classes,
                    pred_instance_labels=pr_inst,
                )
                if mom_stats is not None:
                    momentum_stats.append(mom_stats)

            led_mask = pr_inst != -1
            if led_mask.any():
                ari_scores.append(
                    adjusted_rand_score(gt_inst[led_mask], pr_inst[led_mask])
                )
            else:
                ari_scores.append(float("nan"))

        if comm.get_world_size() > 1:
            gathered = comm.gather((all_stats, ari_scores, momentum_stats), dst=0)
            if comm.get_rank() == 0:
                merged_stats = []
                merged_ari = []
                merged_momentum = []
                for stats_chunk, ari_chunk, momentum_chunk in gathered:
                    merged_stats.extend(stats_chunk)
                    merged_ari.extend(ari_chunk)
                    merged_momentum.extend(momentum_chunk)
                all_stats = merged_stats
                ari_scores = merged_ari
                momentum_stats = merged_momentum
            else:
                self.model.train()
                return

        if not all_stats:
            self.logger.warning("InstanceSegTester: no stats computed")
            self.model.train()
            return

        aggregated = aggregate_instance_results(
            all_stats, require_class_for_match=self.require_class_for_match
        )

        det = aggregated["detection"]
        cls = aggregated["classification_on_matched"]
        det_prec = det["precision"]
        det_rec = det["recall"]
        det_f1 = det["f1"]
        det_iou = det["mean_matched_iou"]
        total_gt = int(det.get("num_gt", 0))
        total_pred = int(det.get("num_pred", 0))
        total_matched = int(det.get("num_matched", 0))
        fp_det = max(total_pred - total_matched, 0)
        fn_det = max(total_gt - total_matched, 0)
        fp_per_gt = (fp_det / total_gt) if total_gt > 0 else 0.0
        fn_per_gt = (fn_det / total_gt) if total_gt > 0 else 0.0

        support = cls["support"]
        if np.any(support > 0):
            precision_macro = float(np.mean(cls["precision"][support > 0]))
            recall_macro = float(np.mean(cls["recall"][support > 0]))
            f1_macro = float(np.mean(cls["f1"][support > 0]))
        else:
            precision_macro = recall_macro = f1_macro = 0.0

        ari_clean = np.asarray(ari_scores, dtype=float)
        ari_clean = ari_clean[~np.isnan(ari_clean)]
        ari_mean = float(np.mean(ari_clean)) if ari_clean.size else float("nan")

        self.logger.info(
            "Detection P={:.3f} R={:.3f} F1={:.3f} IoU={:.3f}".format(
                det_prec, det_rec, det_f1, det_iou
            )
        )
        self.logger.info(
            "Counts GT={} Pred={} TP={} FP={} FN={} (FP/GT={:.3f} FN/GT={:.3f})".format(
                total_gt, total_pred, total_matched, fp_det, fn_det, fp_per_gt, fn_per_gt
            )
        )
        self.logger.info(
            "Classification macro P={:.3f} R={:.3f} F1={:.3f}".format(
                precision_macro, recall_macro, f1_macro
            )
        )
        if not np.isnan(ari_mean):
            self.logger.info("ARI mean={:.3f}".format(ari_mean))
        else:
            self.logger.info("ARI mean=n/a")

        if self.require_class_for_match and "pq" in aggregated:
            pq = aggregated["pq"]
            self.logger.info(
                "Panoptic Quality: PQ={:.3f} RQ={:.3f} SQ={:.3f}".format(
                    pq["PQ"], pq["RQ"], pq["SQ"]
                )
            )

        momentum_aggregated = None
        if momentum_stats:
            momentum_aggregated = self._aggregate_momentum_results(momentum_stats, class_names)
            self._log_momentum_metrics(momentum_aggregated, class_names)

        summary = {
            "det_precision": float(det_prec),
            "det_recall": float(det_rec),
            "det_f1": float(det_f1),
            "det_mean_iou": float(det_iou),
            "total_gt": total_gt,
            "total_pred": total_pred,
            "total_matched": total_matched,
            "fp": fp_det,
            "fn": fn_det,
            "fp_per_gt": float(fp_per_gt),
            "fn_per_gt": float(fn_per_gt),
            "class_precision_macro": float(precision_macro),
            "class_recall_macro": float(recall_macro),
            "class_f1_macro": float(f1_macro),
            "ari_mean": float(ari_mean) if not np.isnan(ari_mean) else None,
        }
        if self.require_class_for_match and "pq" in aggregated:
            summary.update({key.lower(): float(value) for key, value in aggregated["pq"].items()})
        if momentum_aggregated is not None:
            overall = momentum_aggregated["overall"]
            summary.update(
                {
                    "momentum_mae": float(overall["mae"]),
                    "momentum_rmse": float(overall["rmse"]),
                    "momentum_count": int(overall["count"]),
                }
            )

        per_class = []
        for cls_idx, class_name in enumerate(class_names):
            per_class.append(
                {
                    "class_id": cls_idx,
                    "class_name": class_name,
                    "support": int(cls["support"][cls_idx]),
                    "precision": float(cls["precision"][cls_idx]),
                    "recall": float(cls["recall"][cls_idx]),
                    "f1": float(cls["f1"][cls_idx]),
                }
            )
        self.write_eval_artifacts(
            "instance",
            {
                "summary": summary,
                "aggregated": aggregated,
                "momentum": momentum_aggregated,
            },
            [{"row_type": "summary", **summary}]
            + [{"row_type": "per_class", **row} for row in per_class],
        )

        self.logger.info(
            "<<<<<<<<<<<<<<<<< End Panoptic/Instance Segmentation Test <<<<<<<<<<<<<<<<<"
        )
        self.model.train()

    def _eval_momentum(
        self,
        matches,
        pred_instance_momentum,
        momentum_gt,
        gt_instance,
        counts,
        criterion,
        num_classes,
        pred_instance_labels,
    ):
        """Evaluate momentum regression per class for matched instances."""
        try:
            b = 0
            mom_gt_b = criterion._get_batch_tensor(
                momentum_gt, b, counts, torch.device("cpu"), None
            )

            inst_b = gt_instance.squeeze(-1)
            if inst_b.dim() == 0:
                inst_b = inst_b.unsqueeze(0)
            if inst_b.dim() == 2 and inst_b.shape[1] == 1:
                inst_b = inst_b.squeeze(1)
            inst_b = inst_b.cpu()

            unique_gt_ids = torch.unique(inst_b)
            unique_gt_ids = unique_gt_ids[unique_gt_ids >= 0]
            gt_mom_map = {}
            for gt_id in unique_gt_ids:
                mask = inst_b == gt_id
                if mask.any():
                    gt_mom_map[gt_id.item()] = mom_gt_b[mask].float().mean().item()

            if isinstance(pred_instance_labels, np.ndarray):
                pred_inst_t = torch.from_numpy(pred_instance_labels)
            else:
                pred_inst_t = pred_instance_labels

            sorted_ids, sort_idx = torch.sort(pred_inst_t)
            unique_ids_sorted, counts_sorted = torch.unique_consecutive(sorted_ids, return_counts=True)

            cumsum_counts = torch.cat([torch.tensor([0]), counts_sorted.cumsum(0)[:-1]])
            first_indices = sort_idx[cumsum_counts]
            mom_values = pred_instance_momentum[first_indices]

            pr_mom_map = {
                uid.item(): val.item()
                for uid, val in zip(unique_ids_sorted, mom_values)
                if uid.item() >= 0
            }

            matched_preds = []
            for m in matches:
                pid = m["pred_id"]
                gid = m["gt_id"]
                pred_cls = m["pred_cls"]

                if pid in pr_mom_map and gid in gt_mom_map:
                    matched_preds.append(
                        {"pred": pr_mom_map[pid], "gt": gt_mom_map[gid], "cls": pred_cls}
                    )

            if not matched_preds:
                return None

            all_p = np.array([x["pred"] for x in matched_preds])
            all_g = np.array([x["gt"] for x in matched_preds])
            all_c = np.array([x["cls"] for x in matched_preds])
            all_e = all_p - all_g

            class_stats = {}
            for cls_idx in range(num_classes):
                cls_mask = all_c == cls_idx
                if not np.any(cls_mask):
                    continue

                cls_errors = all_e[cls_mask]
                cls_mae = float(np.mean(np.abs(cls_errors)))
                cls_rmse = float(np.sqrt(np.mean(cls_errors ** 2)))
                cls_count = int(np.sum(cls_mask))

                class_stats[cls_idx] = {
                    "mae": cls_mae,
                    "rmse": cls_rmse,
                    "count": cls_count,
                }

            overall_mae = float(np.mean(np.abs(all_e)))
            overall_rmse = float(np.sqrt(np.mean(all_e ** 2)))

            return {
                "per_class": class_stats,
                "overall": {
                    "mae": overall_mae,
                    "rmse": overall_rmse,
                    "count": len(all_e),
                },
            }

        except Exception as e:
            self.logger.warning(f"Error evaluating momentum: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _aggregate_momentum_results(self, momentum_stats_list, class_names):
        """Aggregate momentum evaluation results across batches."""
        num_classes = len(class_names)

        class_mae = {i: [] for i in range(num_classes)}
        class_rmse = {i: [] for i in range(num_classes)}
        class_count = {i: 0 for i in range(num_classes)}

        overall_mae = []
        overall_rmse = []
        overall_count = 0

        for stats in momentum_stats_list:
            if stats is None:
                continue

            for cls_idx, cls_stats in stats["per_class"].items():
                if cls_idx < num_classes:
                    class_mae[cls_idx].append(cls_stats["mae"])
                    class_rmse[cls_idx].append(cls_stats["rmse"])
                    class_count[cls_idx] += cls_stats["count"]

            overall_mae.append(stats["overall"]["mae"])
            overall_rmse.append(stats["overall"]["rmse"])
            overall_count += stats["overall"]["count"]

        aggregated_per_class = {}
        for cls_idx in range(num_classes):
            if len(class_mae[cls_idx]) > 0:
                aggregated_per_class[cls_idx] = {
                    "mae": float(np.mean(class_mae[cls_idx])),
                    "rmse": float(np.mean(class_rmse[cls_idx])),
                    "count": class_count[cls_idx],
                }

        aggregated_overall = {
            "mae": float(np.mean(overall_mae)) if overall_mae else 0.0,
            "rmse": float(np.mean(overall_rmse)) if overall_rmse else 0.0,
            "count": overall_count,
        }

        return {
            "per_class": aggregated_per_class,
            "overall": aggregated_overall,
        }

    def _log_momentum_metrics(self, momentum_aggregated, class_names):
        """Log momentum regression metrics."""
        self.logger.info("Momentum Regression Metrics:")

        overall = momentum_aggregated["overall"]
        self.logger.info(
            "Overall: MAE={:.4f} RMSE={:.4f} Count={}".format(
                overall["mae"], overall["rmse"], overall["count"]
            )
        )

        per_class = momentum_aggregated["per_class"]
        if per_class:
            self.logger.info("Per-class metrics:")
            for cls_idx in sorted(per_class.keys()):
                cls_name = (
                    class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                )
                cls_stats = per_class[cls_idx]
                self.logger.info(
                    "  {}: MAE={:.4f} RMSE={:.4f} Count={}".format(
                        cls_name, cls_stats["mae"], cls_stats["rmse"], cls_stats["count"]
                    )
                )

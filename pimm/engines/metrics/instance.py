"""Instance segmentation metric computation shared by evaluator and tester."""

from __future__ import annotations

import numpy as np


def eval_instances(
    gt_inst,  # (N,) int, ground-truth instance id per point
    pr_inst,  # (N,) int, predicted instance id per point
    gt_pid,  # either (N,) int per-point PID (constant within GT instance) OR (M_gt,) int table indexed by inst id
    pr_pid,  # either (N,) int per-point PID (constant within Pred instance) OR (M_pr,) int table indexed by inst id
    class_names=("photon", "electron", "muon", "pion", "proton"),
    iou_thresh=0.5,
    require_class_for_match=False,  # True means panoptic-style class-aware matching
):
    """
    Returns:
    {
        "detection": {precision, recall, f1, num_gt, num_pred, num_matched, mean_matched_iou, iou_thresh, matching},
        "classification_on_matched": {precision[K], recall[K], f1[K], confusion[KxK], support[K], class_names},
        "pq": {PQ, RQ, SQ}  # only when require_class_for_match=True
        "matches": [ {pred_id, gt_id, iou, pred_cls, gt_cls, intersection, pred_size, gt_size}, ... ]
    }
    """
    K = len(class_names)
    N = gt_inst.shape[0]
    assert pr_inst.shape[0] == N

    # Filter out background and invalid PIDs (bg_id=-1, pid=5)
    valid_mask = (gt_inst >= 0) & (pr_inst >= 0)

    # sizes per instance (counts of points)
    pr_ids, pr_sizes = np.unique(pr_inst[pr_inst >= 0], return_counts=True)
    gt_ids, gt_sizes = np.unique(gt_inst[gt_inst >= 0], return_counts=True)
    pr_size = {int(i): int(c) for i, c in zip(pr_ids, pr_sizes)}
    gt_size = {int(i): int(c) for i, c in zip(gt_ids, gt_sizes)}

    # intersections via pair counting in O(N)
    if valid_mask.any():
        pairs = np.stack([pr_inst[valid_mask], gt_inst[valid_mask]], axis=1)
        uniq_pairs, inter_counts = np.unique(pairs, axis=0, return_counts=True)
        # mapping (p,g) -> inter
        inter_map = {
            (int(p), int(g)): int(c) for (p, g), c in zip(uniq_pairs, inter_counts)
        }
    else:
        inter_map = {}

    # build per-instance PID from arrays
    def build_inst_pid_map(inst_ids_present, inst_ids_per_point, pid_array):
        inst_ids_present = list(map(int, inst_ids_present))
        m = int(np.max(inst_ids_present)) + 1 if inst_ids_present else 0

        # case A: pid_array is per-instance table (len > max inst id)
        if pid_array.ndim == 1 and pid_array.shape[0] >= m and m > 0:
            return {i: int(pid_array[i]) for i in inst_ids_present if 0 <= int(pid_array[i]) < K}

        # case B: pid_array is per-point; take the mode for each instance
        pid_map = {}
        if len(inst_ids_present) == 0:
            return pid_map
        # restrict to points with valid instances and PIDs
        use = (inst_ids_per_point >= 0) & (pid_array >= 0) & (pid_array < K)
        inst_vals = inst_ids_per_point[use].astype(np.int64)
        pid_vals = pid_array[use].astype(np.int64)
        # count (inst, pid)
        ip = np.stack([inst_vals, pid_vals], axis=1)
        uniq_ip, cnts = np.unique(ip, axis=0, return_counts=True)
        # for each inst, choose pid with max count
        # sort by inst then count desc
        order = np.lexsort(
            (-cnts, uniq_ip[:, 0])
        )  # primary: inst asc, secondary: count desc
        uniq_ip_sorted = uniq_ip[order]
        cnts_sorted = cnts[order]
        # first occurrence per inst in this order is the mode
        _, first_idx = np.unique(uniq_ip_sorted[:, 0], return_index=True)
        for idx in first_idx:
            inst_i = int(uniq_ip_sorted[idx, 0])
            pid_i = int(uniq_ip_sorted[idx, 1])
            if 0 <= pid_i < K:  # Only include valid PIDs
                pid_map[inst_i] = pid_i
        return pid_map

    gt_pid_map = build_inst_pid_map(gt_ids, gt_inst, gt_pid)
    pr_pid_map = build_inst_pid_map(pr_ids, pr_inst, pr_pid)

    # optional class-aware gating for matching
    def class_ok(p_id, g_id):
        if not require_class_for_match:
            return True
        return pr_pid_map.get(p_id, -999) == gt_pid_map.get(g_id, -998)

    # candidate pairs with IoU
    cand = []
    for (p, g), inter in inter_map.items():
        if p not in pr_size or g not in gt_size:
            continue
        if not class_ok(p, g):
            continue
        union = pr_size[p] + gt_size[g] - inter
        if union <= 0:
            continue
        iou = inter / union
        cand.append((iou, p, g, inter))

    # greedy one-to-one matching by IoU
    cand.sort(reverse=True, key=lambda t: t[0])
    used_p, used_g = set(), set()
    matches = []
    for iou, p, g, inter in cand:
        if iou < iou_thresh:
            break
        if p in used_p or g in used_g:
            continue
        matches.append((p, g, iou, inter))
        used_p.add(p)
        used_g.add(g)

    num_gt = len(gt_ids)
    num_pred = len(pr_ids)
    num_matched = len(matches)
    fp = num_pred - num_matched
    fn = num_gt - num_matched

    det_prec = num_matched / (num_matched + fp) if (num_matched + fp) else 0.0
    det_rec = num_matched / (num_matched + fn) if (num_matched + fn) else 0.0
    det_f1 = (
        (2 * det_prec * det_rec / (det_prec + det_rec)) if (det_prec + det_rec) else 0.0
    )
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

    # classification on matched pairs
    K = len(class_names)
    confusion = np.zeros((K, K), dtype=int)
    out_matches = []
    iou_sum_for_pq = 0.0
    tp_for_pq = 0
    for p, g, iou, inter in matches:
        pred_cls = pr_pid_map.get(p, -1)
        gt_cls = gt_pid_map.get(g, -1)
        if 0 <= gt_cls < K and 0 <= pred_cls < K:
            confusion[gt_cls, pred_cls] += 1
        out_matches.append(
            {
                "pred_id": p,
                "gt_id": g,
                "iou": float(iou),
                "pred_cls": int(pred_cls),
                "gt_cls": int(gt_cls),
                "intersection": int(inter),
                "pred_size": pr_size[p],
                "gt_size": gt_size[g],
            }
        )
        if require_class_for_match and pred_cls == gt_cls and 0 <= pred_cls < K:
            tp_for_pq += 1
            iou_sum_for_pq += iou

    support = np.array([confusion[i, :].sum() for i in range(K)], dtype=int)
    precision = np.zeros(K)
    recall = np.zeros(K)
    f1 = np.zeros(K)
    for i in range(K):
        tp = confusion[i, i]
        fp_c = confusion[:, i].sum() - tp
        fn_c = confusion[i, :].sum() - tp
        pr = tp / (tp + fp_c) if (tp + fp_c) else 0.0
        rc = tp / (tp + fn_c) if (tp + fn_c) else 0.0
        precision[i], recall[i] = pr, rc
        f1[i] = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0

    out = {
        "detection": {
            "precision": det_prec,
            "recall": det_rec,
            "f1": det_f1,
            "num_gt": num_gt,
            "num_pred": num_pred,
            "num_matched": num_matched,
            "mean_matched_iou": mean_iou,
            "iou_thresh": iou_thresh,
            "matching": "class-aware" if require_class_for_match else "unlabeled",
        },
        "classification_on_matched": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": confusion,
            "support": support,
            "class_names": list(class_names),
        },
        "matches": out_matches,
    }

    if require_class_for_match:
        # PQ = SQ * RQ, with SQ = mean IoU over TPs (class-correct)
        # RQ = TP / (TP + 0.5 FP + 0.5 FN)
        sq = (iou_sum_for_pq / tp_for_pq) if tp_for_pq else 0.0
        rq_den = tp_for_pq + 0.5 * fp + 0.5 * fn
        rq = (tp_for_pq / rq_den) if rq_den > 0 else 0.0
        out["pq"] = {"PQ": rq * sq, "RQ": rq, "SQ": sq}

    return out

def aggregate_instance_results(stats_list, require_class_for_match=False):
    """Pool counts across events to compute global micro metrics and confusion."""
    if len(stats_list) == 0:
        raise ValueError("stats_list is empty")

    # Assume consistent class set across events
    cls0 = stats_list[0]["classification_on_matched"]
    class_names = list(cls0["class_names"])
    K = len(class_names)

    total_gt = total_pred = total_matched = 0
    # IoU sums
    iou_sum_all_matches = 0.0
    iou_sum_pq = 0.0  # class-correct only (for PQ)
    tp_pq = 0  # class-correct matched count

    # pooled confusion/support
    pooled_conf = np.zeros((K, K), dtype=int)
    pooled_support = np.zeros(K, dtype=int)

    for s in stats_list:
        det = s["detection"]
        total_gt += det["num_gt"]
        total_pred += det["num_pred"]
        total_matched += det["num_matched"]

        # accumulate IoUs from matches
        for m in s["matches"]:
            iou_sum_all_matches += float(m["iou"])
            if 0 <= m["gt_cls"] < K and m["gt_cls"] == m["pred_cls"]:
                tp_pq += 1
                iou_sum_pq += float(m["iou"])

        cls = s["classification_on_matched"]
        pooled_conf += np.asarray(cls["confusion"], dtype=int)
        pooled_support += np.asarray(cls["support"], dtype=int)

    # Detection micro metrics
    tp_det = total_matched
    fp_det = total_pred - total_matched
    fn_det = total_gt - total_matched

    det_prec = tp_det / (tp_det + fp_det) if (tp_det + fp_det) else 0.0
    det_rec = tp_det / (tp_det + fn_det) if (tp_det + fn_det) else 0.0
    det_f1 = (
        (2 * det_prec * det_rec / (det_prec + det_rec)) if (det_prec + det_rec) else 0.0
    )
    mean_iou = (iou_sum_all_matches / tp_det) if tp_det else 0.0

    # Classification micro-from-pooled-confusion
    tp = np.diag(pooled_conf)
    fp = pooled_conf.sum(axis=0) - tp
    fn = pooled_conf.sum(axis=1) - tp

    precision = np.zeros(K)
    recall = np.zeros(K)
    f1 = np.zeros(K)
    for i in range(K):
        pr = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        rc = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        precision[i], recall[i] = pr, rc
        f1[i] = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0

    # Build a "global" result dict in the same schema
    res_global = {
        "detection": {
            "precision": det_prec,
            "recall": det_rec,
            "f1": det_f1,
            "num_gt": int(total_gt),
            "num_pred": int(total_pred),
            "num_matched": int(total_matched),
            "mean_matched_iou": float(mean_iou),
            "iou_thresh": stats_list[0]["detection"]["iou_thresh"],
            "matching": stats_list[0]["detection"]["matching"],
        },
        "classification_on_matched": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": pooled_conf,
            "support": pooled_support,
            "class_names": class_names,
        },
        "matches": [],  # not retained at global level
    }

    # Global PQ if class-aware matching was used
    if require_class_for_match:
        rq_den = tp_pq + 0.5 * fp_det + 0.5 * fn_det
        rq = (tp_pq / rq_den) if rq_den > 0 else 0.0
        sq = (iou_sum_pq / tp_pq) if tp_pq > 0 else 0.0
        res_global["pq"] = {"PQ": rq * sq, "RQ": rq, "SQ": sq}

    return res_global

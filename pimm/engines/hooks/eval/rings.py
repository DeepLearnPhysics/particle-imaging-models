"""Overlap-aware evaluator for the Water-Cherenkov ring panoptic detector.

Standard panoptic/instance evaluators assume a per-point partition (each point
in exactly one instance) and compare integer instance-id maps. Cherenkov rings
overlap, so this hook instead matches predicted ring *masks* to GT ring masks one
to one by mask IoU (``match_rings``)
and reports, on the held-out split:

  * detection precision / recall / F1 and mean matched IoU at an IoU threshold,
  * PID accuracy on matched rings,
  * charge-weighted detection and particle F1 (sites counted by the rings' PE),
  * on the charge-weighted matches: optional energy and required momentum MAE
    (log-target space) plus median fractional error, direction angular error
    (degrees) and vertex L2 (normalized + meters),
  * electron/muon vertex and relative-momentum 68% containment.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import pimm.utils.comm as comm
from pimm.distributed import unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase, get_writer_step
from pimm.observability import structured_logger as sl


def _finite_array(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _nanmean(values):
    values = _finite_array(values)
    return float(values.mean()) if values.size else float("nan")


def _nanmedian(values):
    values = _finite_array(values)
    return float(np.median(values)) if values.size else float("nan")


def _q68(values):
    """68% containment of the absolute residual."""
    values = np.abs(_finite_array(values))
    return float(np.quantile(values, 0.68)) if values.size else float("nan")


def _log_fractional_error(pred_log, true_log):
    """Return ``|p_pred - p_true| / p_true`` for log-space predictions."""
    delta_log = float(pred_log) - float(true_log)
    return abs(float(np.expm1(delta_log))) if abs(delta_log) < 20 else float("nan")


def match_rings(pred, truth, weight=None, iou_thresh=0.5):
    """Pred-truth pairs matched one to one at IoU >= ``iou_thresh`` (maximum matching over those pairs, ties broken by
    the highest summed IoU), and the IoU matrix; pred (K, N), truth (T, N) bool. With ``weight`` (N,), each site counts
    by it (charge-weighted IoU)."""
    if not len(pred) or not len(truth):
        return np.zeros(0, int), np.zeros(0, int), np.zeros((len(pred), len(truth)))
    w = (
        torch.ones(pred.shape[1], device=pred.device)
        if weight is None
        else weight.float()
    )
    inter = (pred.float() * w) @ truth.float().T
    size_p, size_t = pred.float() @ w, truth.float() @ w
    iou = (
        (inter / (size_p[:, None] + size_t[None] - inter).clamp(min=1e-6)).cpu().numpy()
    )
    # one point per match, plus an IoU bonus that sums to under one match: never trades a match for IoU
    gain = (iou >= iou_thresh) * (1 + iou / (min(iou.shape) + 1))
    rows, cols = linear_sum_assignment(-gain)
    keep = iou[rows, cols] >= iou_thresh
    return rows[keep], cols[keep], iou


def ring_residuals(ring, direction, momentum, vertex, vertex_known, coord_scale):
    """A matched ring's kinematic residuals: direction error (degrees), |Δp| / p_true from the log momenta, and vertex
    distance in metres (NaN when the true vertex is unknown)."""
    d = F.normalize(torch.as_tensor(ring["direction"]).float().reshape(3).cpu(), dim=0)
    angle = float(
        torch.rad2deg(torch.arccos((d @ direction.float().cpu()).clamp(-1, 1)))
    )
    delta = float(
        torch.as_tensor(ring["momentum"]).float().reshape(-1)[0].cpu() - momentum.cpu()
    )
    momentum_frac = abs(float(np.expm1(delta))) if abs(delta) < 20 else float("nan")
    gap = (
        torch.as_tensor(ring["vertex"]).float().reshape(3).cpu() - vertex.float().cpu()
    ).norm() * coord_scale
    return angle, momentum_frac, float(gap) if vertex_known else float("nan")


@HOOKS.register_module()
class RingPanopticEvaluator(HookBase):
    """Overlap-aware evaluator for the Water-Cherenkov ring panoptic detector.

    Unlike per-point partition evaluators, this hook matches predicted ring
    *masks* to ground-truth ring masks one to one by mask IoU (``match_rings``:
    maximum matching over the pairs at IoU >= ``iou_thresh``; rings overlap, so a
    point may belong to several rings). For each validation event it
    postprocesses predictions via either the dedicated ring detector's
    ``model.postprocess_event`` or a unified detector's overlap-preserving
    ``model.postprocess`` and, at the
    configured IoU threshold, reports detection precision/recall/F1 and mean
    matched IoU, PID accuracy on matched rings, energy/momentum MAE in
    log-target space plus median fractional error when the corresponding query
    head is configured, direction angular error
    (degrees), and vertex L2 in normalized units and meters. A second matching
    weights each site by the rings' PE (``inst_pe``) and gives the charge-weighted
    ``q_det_f1`` and ``q_particle_f1`` (2 x matches with the right PID / (predicted
    + true rings)); the kinematics are computed on these matches (``ring_residuals``,
    as the WC scorer). For matched true
    electrons and muons it also reports vertex-distance q68 and relative-momentum
    q68. Counts and
    error lists are gathered to rank 0 on multi-GPU. ``selection_metric`` is
    published as the checkpoint-selection metric (``current_metric_value``,
    ``current_metric_name`` = ``<prefix>/<selection_metric>``) when ``set_current_metric``.
    Runs after every step when ``every_n_steps > 0`` (when
    ``(global_iter + 1) % every_n_steps == 0``), otherwise after each epoch; only
    when ``cfg.evaluate`` is true. Registered as ``RingPanopticEvaluator`` (use
    as ``type`` in a ``hooks=[...]`` entry).

    Args:
        every_n_steps (int): Step cadence; ``0`` evaluates once per epoch.
            Defaults to ``0``.
        iou_thresh (float): Minimum mask IoU for a matched prediction to count as
            a true positive. Defaults to ``0.5``.
        conf_threshold (float): Ring confidence threshold for postprocessing.
            Defaults to ``0.5``.
        mask_threshold (float): Per-point probability threshold for binarizing
            predicted ring masks. Defaults to ``0.5``.
        min_points (int): Minimum points for a predicted ring to be kept.
            Defaults to ``5``.
        coord_scale (float): Normalized-to-meters scale used to report vertex
            L2 in meters. Defaults to ``18.1``.
        particle_classes (tuple[int, int]): Electron and muon class indices for
            particle-specific residuals. Defaults to ``(0, 2)``.
        prefix (str): Writer metric prefix (leading/trailing slashes stripped).
            Defaults to ``"val"``.
        max_batches (int): Cap on validation batches per eval; ``0`` uses all.
            Defaults to ``0``.
        set_current_metric (bool): Publish ``selection_metric`` as the
            checkpoint-selection metric. Defaults to ``True``.
        selection_metric (str): The logged metric that selects checkpoints, e.g.
            ``"q_particle_f1"``. Defaults to ``"det_f1"``.
        dedup_iou (float, optional): Unified detector only: the overlap
            postprocess's near-duplicate removal threshold (``1.0`` keeps every
            ring). Defaults to ``None`` (the model's setting).

    Vertex residuals skip rings whose ``ring_vertex_valid`` is false (their true
    vertex is unknown).

    Note:
        The selection metric is higher-is-better (ring detection F1 by default). Energy is
        optional; momentum is expected for the RPS ring task. Both are evaluated
        in log-target space when present, and their fractional-error columns are
        medians that skip pathological (``|Δlog| >= 20``) matches. Evaluation
        no-ops when ``val_loader`` is ``None``.

    Example:
        Add to ``cfg.hooks`` for the Cherenkov ring detector; it Hungarian-matches
        predicted ring masks to GT and sets the selection metric:

        .. code-block:: python

            hooks = [
                dict(type="RingPanopticEvaluator", every_n_steps=1000,
                     iou_thresh=0.5, prefix="val"),
            ]
    """

    def __init__(
        self,
        every_n_steps=0,
        iou_thresh=0.5,
        conf_threshold=0.5,
        mask_threshold=0.5,
        min_points=5,
        coord_scale=18.1,
        prefix="val",
        max_batches=0,
        set_current_metric=True,
        dedup_iou=None,
        selection_metric="det_f1",
        particle_classes=(0, 2),
    ):
        self.every_n_steps = int(every_n_steps)
        self.iou_thresh = float(iou_thresh)
        self.conf_threshold = float(conf_threshold)
        self.mask_threshold = float(mask_threshold)
        self.min_points = int(min_points)
        self.coord_scale = float(coord_scale)
        self.prefix = prefix.strip("/")
        self.max_batches = int(max_batches)
        self.set_current_metric = bool(set_current_metric)
        self.dedup_iou = dedup_iou  # unified detector: near-duplicate removal (None: the model's; 1.0: none)
        self.selection_metric = str(selection_metric)
        self.particle_classes = tuple(int(pid) for pid in particle_classes)
        if len(self.particle_classes) != 2:
            raise ValueError("particle_classes must contain electron and muon indices")

    def after_step(self):
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            gi = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"]
                * self.trainer.comm_info["epoch"]
            )
            if (gi + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def _particle_regression_metrics(self, matched_pids, momentum_frac, vertex_m, direction_deg):
        """Summarize matched true-electron/muon residuals only (vertex distances in metres, directions in degrees)."""
        if not (len(matched_pids) == len(momentum_frac) == len(vertex_m) == len(direction_deg)):
            raise ValueError(
                "matched PID, momentum, vertex and direction residual lists must align"
            )

        metrics = {}
        for particle, pid in zip(("e", "mu"), self.particle_classes):
            select = [
                idx for idx, true_pid in enumerate(matched_pids) if true_pid == pid
            ]
            particle_momentum = [momentum_frac[idx] for idx in select]
            particle_vertex = [vertex_m[idx] for idx in select]
            vertex_mean = _nanmean(particle_vertex)
            vertex_q68 = _q68(particle_vertex)
            metrics.update(
                {
                    f"vertex_{particle}_l2_norm": vertex_mean / self.coord_scale,
                    f"vertex_{particle}_l2_m": vertex_mean,
                    f"vertex_{particle}_q68_norm": vertex_q68 / self.coord_scale,
                    f"vertex_{particle}_q68_m": vertex_q68,
                    f"momentum_{particle}_frac_median": _nanmedian(particle_momentum),
                    f"momentum_{particle}_frac_q68": _q68(particle_momentum),
                    f"direction_{particle}_q68_deg": _q68([direction_deg[idx] for idx in select]),
                }
            )
        return metrics

    @staticmethod
    def _split_overlap_targets(input_dict, batch_size, membership_threshold=0.0):
        """Build per-event ring targets from the dataset's padded overlap GT."""
        required = ("offset", "inst_pe", "ring_valid", "ring_pid")
        missing = [key for key in required if key not in input_dict]
        if missing:
            raise KeyError(
                "unified ring evaluation needs overlap targets: " + ", ".join(missing)
            )
        if batch_size == 0:
            return []

        valid_all = input_dict["ring_valid"].reshape(-1)
        assert (
            valid_all.numel() % batch_size == 0
        ), "ring_valid must contain one fixed-width ring block per event"
        max_rings = valid_all.numel() // batch_size
        offsets = input_dict["offset"].reshape(-1)
        assert (
            offsets.numel() == batch_size
        ), "offset must contain one cumulative point count per event"

        targets = []
        point_start = 0
        for batch_idx in range(batch_size):
            point_end = int(offsets[batch_idx])
            ring_start = batch_idx * max_rings
            ring_end = ring_start + max_rings
            valid = valid_all[ring_start:ring_end].bool()
            membership = input_dict["inst_pe"][point_start:point_end]
            target = {
                "masks": (membership[:, valid] > float(membership_threshold)).T.float(),
                "pid": input_dict["ring_pid"][ring_start:ring_end][valid].long(),
                "charge": membership[:, valid]
                .float()
                .sum(1),  # the rings' PE at each point
            }
            for name in ("energy", "momentum", "vertex", "direction"):
                key = f"ring_{name}"
                if key in input_dict:
                    target[name] = input_dict[key][ring_start:ring_end][valid].float()
            if (
                "ring_vertex_valid" in input_dict
            ):  # rings whose true vertex is unknown are left out of the vertex metrics
                target["vertex_valid"] = input_dict["ring_vertex_valid"][
                    ring_start:ring_end
                ][valid].bool()
            targets.append(target)
            point_start = point_end
        return targets

    @sl.log_trace_span("evaluation.ring_panoptic")
    def eval(self):
        if self.trainer.val_loader is None:
            return
        sl.add_step_tag("evaluation")
        self.trainer.logger.info(
            ">>>>>>>>>>>>>> Start Ring Panoptic Evaluation >>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()
        model = unwrap_model(self.trainer.model)

        n_gt = n_pred = n_tp = q_tp = 0
        iou_sum = 0.0
        pid_correct = pid_total = q_pid_correct = 0
        e_abs, e_frac, m_abs, m_frac = [], [], [], []
        dir_deg, vtx_m, matched_pids = [], [], []

        for bi, input_dict in enumerate(self.trainer.val_loader):
            if self.max_batches and bi >= self.max_batches:
                break
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.cuda(non_blocking=True)
            with torch.no_grad():
                out = self.trainer.model(input_dict)
            rings_by_event = model.postprocess(
                out,
                conf_threshold=self.conf_threshold,
                mask_threshold=self.mask_threshold,
                min_points=self.min_points,
                **({} if self.dedup_iou is None else {"dedup_iou": self.dedup_iou}),
            )
            gts = self._split_overlap_targets(input_dict, batch_size=len(rings_by_event))

            for rings, gt in zip(rings_by_event, gts):
                R = gt["masks"].shape[0]
                n_gt += R
                n_pred += len(rings)
                if R == 0 or len(rings) == 0:
                    continue
                pm = torch.stack([r["mask"] for r in rings]).to(gt["masks"].device)
                truth = gt["masks"].bool()
                rows, cols, iou = match_rings(
                    pm, truth, iou_thresh=self.iou_thresh
                )  # (P, R) IoU
                n_tp += len(rows)
                iou_sum += float(iou[rows, cols].sum())
                pid_total += len(rows)
                pid_correct += sum(
                    int(rings[p]["pid"] == int(gt["pid"][g]))
                    for p, g in zip(rows, cols)
                )
                # charge-weighted matches: the particle list, and its kinematics
                charge = gt["charge"]
                q_rows, q_cols, _ = match_rings(pm, truth, charge, self.iou_thresh)
                for p, g in zip(q_rows, q_cols):
                    q_tp += 1
                    r = rings[p]
                    true_pid = int(gt["pid"][g])
                    q_pid_correct += int(r["pid"] == true_pid)
                    matched_pids.append(true_pid)
                    if "energy" in gt and "energy" in r:
                        eg = float(gt["energy"][g])
                        ep = float(r["energy"])
                        e_abs.append(abs(ep - eg))
                        e_frac.append(_log_fractional_error(ep, eg))
                    m_abs.append(abs(float(r["momentum"]) - float(gt["momentum"][g])))
                    known = "vertex_valid" not in gt or bool(gt["vertex_valid"][g])
                    angle, frac, gap = ring_residuals(
                        r,
                        gt["direction"][g],
                        gt["momentum"][g],
                        gt["vertex"][g],
                        known,
                        self.coord_scale,
                    )
                    dir_deg.append(angle)
                    m_frac.append(frac)
                    vtx_m.append(gap)

        # aggregate across ranks
        if comm.get_world_size() > 1:
            payload = (
                n_gt,
                n_pred,
                n_tp,
                iou_sum,
                pid_correct,
                pid_total,
                e_abs,
                e_frac,
                m_abs,
                m_frac,
                dir_deg,
                vtx_m,
                matched_pids,
                q_tp,
                q_pid_correct,
            )
            gathered = comm.gather(payload, dst=0)
            if comm.get_rank() != 0:
                self.trainer.model.train()
                return
            n_gt = sum(g[0] for g in gathered)
            n_pred = sum(g[1] for g in gathered)
            n_tp = sum(g[2] for g in gathered)
            iou_sum = sum(g[3] for g in gathered)
            pid_correct = sum(g[4] for g in gathered)
            pid_total = sum(g[5] for g in gathered)
            e_abs = sum((g[6] for g in gathered), [])
            e_frac = sum((g[7] for g in gathered), [])
            m_abs = sum((g[8] for g in gathered), [])
            m_frac = sum((g[9] for g in gathered), [])
            dir_deg = sum((g[10] for g in gathered), [])
            vtx_m = sum((g[11] for g in gathered), [])
            matched_pids = sum((g[12] for g in gathered), [])
            q_tp = sum(g[13] for g in gathered)
            q_pid_correct = sum(g[14] for g in gathered)

        prec = n_tp / max(n_pred, 1)
        rec = n_tp / max(n_gt, 1)
        f1 = 2 * n_tp / max(n_pred + n_gt, 1)
        miou = iou_sum / max(n_tp, 1)
        pid_acc = pid_correct / max(pid_total, 1)

        metrics = {
            "det_precision": prec,
            "det_recall": rec,
            "det_f1": f1,
            "mean_matched_iou": miou,
            "pid_acc": pid_acc,
            "q_det_f1": 2 * q_tp / max(n_pred + n_gt, 1),
            "q_particle_f1": 2 * q_pid_correct / max(n_pred + n_gt, 1),
            "energy_mae_log": _nanmean(e_abs),
            "energy_frac_median": _nanmedian(e_frac),
            "momentum_mae_log": _nanmean(m_abs),
            "momentum_frac_median": _nanmedian(m_frac),
            "direction_deg": _nanmean(dir_deg),
            "vertex_l2_norm": _nanmean(vtx_m) / self.coord_scale,
            "vertex_l2_m": _nanmean(vtx_m),
        }
        metrics.update(self._particle_regression_metrics(matched_pids, m_frac, vtx_m, dir_deg))

        self.trainer.logger.info(
            "[ring] GT=%d Pred=%d TP=%d | P=%.3f R=%.3f F1=%.3f IoU=%.3f | "
            "PIDacc=%.3f | charge-weighted F1=%.3f particle F1=%.3f"
            % (
                n_gt,
                n_pred,
                n_tp,
                prec,
                rec,
                f1,
                miou,
                pid_acc,
                metrics["q_det_f1"],
                metrics["q_particle_f1"],
            )
        )
        self.trainer.logger.info(
            "[ring] E_mae(log)=%.3f (frac~%.2f) | p_mae(log)=%.3f (frac~%.2f) | "
            "dir=%.1fdeg | vtx=%.3f (%.2fm)"
            % (
                metrics["energy_mae_log"],
                metrics["energy_frac_median"],
                metrics["momentum_mae_log"],
                metrics["momentum_frac_median"],
                metrics["direction_deg"],
                metrics["vertex_l2_norm"],
                metrics["vertex_l2_m"],
            )
        )
        for particle in ("e", "mu"):
            vertex_key = f"vertex_{particle}_q68_m"
            momentum_key = f"momentum_{particle}_frac_q68"
            if vertex_key in metrics:
                self.trainer.logger.info(
                    "[ring][%s] vertex q68=%.2fm | |delta p|/p_true q68=%.3f"
                    % (particle, metrics[vertex_key], metrics[momentum_key])
                )

        if self.trainer.writer is not None:
            step = get_writer_step(self.trainer)
            for k, v in metrics.items():
                if np.isfinite(v):
                    self.trainer.writer.add_scalar(f"{self.prefix}/{k}", v, step)

        if self.set_current_metric:
            value, name = metrics[self.selection_metric], self.selection_metric
            if "_q68" in name:  # an error, lower is better; the checkpoint saver keeps the highest value
                value, name = -value, f"neg_{name}"
            self.trainer.comm_info["current_metric_value"] = value
            self.trainer.comm_info["current_metric_name"] = f"{self.prefix}/{name}"

        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<<< End Ring Panoptic Evaluation <<<<<<<<<<<<<<<<<"
        )
        self.trainer.model.train()

"""Water-Cherenkov probes, physical resolutions, and event selections."""

from __future__ import annotations

import numpy as np
import json
import os
import torch
import pimm.utils.comm as comm
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase


def _q68(x):
    """68% containment (the standard fiTQun resolution quote)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.quantile(np.abs(x), 0.68)) if x.size else float("nan")


def pid_metrics(y_true, y_score):
    """Binary PID. ``y_true`` in {0,1}; ``y_score`` = score for class 1.

    Returns accuracy, ROC AUC, signal efficiency at 99% / 99.9% background
    rejection (the way nu_e-appearance analyses quote it), and background
    mis-ID at 90% / 99% signal efficiency.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    out = {"n": int(y_true.size)}
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return dict(out, accuracy=float("nan"), auc=float("nan"))

    # accuracy at the natural 0.5-probability / 0-logit threshold
    thr = 0.5 if (y_score.min() >= 0.0 and y_score.max() <= 1.0) else 0.0
    out["accuracy"] = float(np.mean((y_score >= thr).astype(int) == y_true))

    # ROC AUC via rank statistic (Mann-Whitney U), no sklearn dependency
    # average ranks for ties
    _, inv, cnt = np.unique(y_score, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = csum - (cnt - 1) / 2.0
    ranks = avg[inv]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(y_true.size - n_pos)
    auc = (np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    out["auc"] = float(auc)

    # signal (class-1) efficiency at fixed background (class-0) rejection
    neg = np.sort(y_score[y_true == 0])
    pos = y_score[y_true == 1]
    for rej in (0.99, 0.999):
        cut = np.quantile(neg, rej)  # rej fraction of bkg below this score
        out[f"sig_eff@{rej:g}bkgrej"] = float(np.mean(pos >= cut))

    # background (class-0) mis-ID at fixed signal (class-1) efficiency
    for eff in (0.9, 0.99):
        cut = np.quantile(
            pos, 1.0 - eff
        )  # eff fraction of signal at or above this score
        out[f"bkg_misid@{eff:g}sigeff"] = float(np.mean(neg >= cut))
    return out


def energy_resolution(true_e, pred_e):
    """Relative energy resolution in %. ``rel = (reco - true)/true``."""
    true_e = np.asarray(true_e, dtype=np.float64).reshape(-1)
    pred_e = np.asarray(pred_e, dtype=np.float64).reshape(-1)
    rel = (pred_e - true_e) / np.clip(np.abs(true_e), 1e-9, None)
    return {
        "n": int(true_e.size),
        "bias_pct": 100.0 * float(np.mean(rel)),
        "res_pct": 100.0 * float(np.std(rel)),
        "q68_pct": 100.0 * _q68(rel),
    }


def vertex_resolution(true_xyz, pred_xyz, direction=None):
    """Vertex resolution in cm (inputs in meters).

    Reports mean/median/68%-containment of the 3D miss distance. If a true
    ``direction`` (N,3 unit) is given, also splits into longitudinal (along the
    track) and transverse components, as fiTQun does.
    """
    t = np.asarray(true_xyz, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(pred_xyz, dtype=np.float64).reshape(-1, 3)
    diff_cm = (p - t) * 100.0
    d = np.linalg.norm(diff_cm, axis=1)
    out = {
        "n": int(t.shape[0]),
        "mean_cm": float(np.mean(d)),
        "median_cm": float(np.median(d)),
        "q68_cm": float(np.quantile(d, 0.68)),
    }
    if direction is not None:
        u = np.asarray(direction, dtype=np.float64).reshape(-1, 3)
        u = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-9, None)
        longi = np.sum(diff_cm * u, axis=1)
        trans = np.linalg.norm(diff_cm - longi[:, None] * u, axis=1)
        out["long_q68_cm"] = float(np.quantile(np.abs(longi), 0.68))
        out["trans_q68_cm"] = float(np.quantile(trans, 0.68))
    return out


def angular_resolution(true_dir, pred_dir):
    """Angular error in degrees between unit direction vectors."""
    t = np.asarray(true_dir, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(pred_dir, dtype=np.float64).reshape(-1, 3)
    t = t / np.clip(np.linalg.norm(t, axis=1, keepdims=True), 1e-9, None)
    p = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-9, None)
    cos = np.clip(np.sum(t * p, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return {
        "n": int(t.shape[0]),
        "mean_deg": float(np.mean(ang)),
        "median_deg": float(np.median(ang)),
        "q68_deg": float(np.quantile(ang, 0.68)),
        "cos_mean": float(np.mean(cos)),
    }


# Detector half-extents in meters (cylinder). Verified from sensor_positions.
DET_RADIUS_M = 16.9
DET_HALF_Z_M = 18.1


def fiducial_mask(vertex, margin_m=1.0, radius_m=DET_RADIUS_M, half_z_m=DET_HALF_Z_M):
    """Cylindrical fiducial cut: vertex at least ``margin_m`` from every wall.

    ``vertex`` is (N, 3) in meters. Returns an (N,) bool mask.
    """
    vertex = np.asarray(vertex, dtype=np.float64).reshape(-1, 3)
    r_xy = np.hypot(vertex[:, 0], vertex[:, 1])
    return (r_xy <= radius_m - margin_m) & (np.abs(vertex[:, 2]) <= half_z_m - margin_m)


def head_metrics(kind, pred, truth, label=None):
    """fitqun_metrics of one head's val predictions, plus its selection score (higher is better): PID AUC (pred: the
    signal logit), and minus the vertex q68 (cm), direction q68 (deg) or momentum |dp / p| q68 (%; pred: log(|p| /
    1 GeV/c))."""
    if kind == "pid":
        m = pid_metrics(label, pred)
        return dict(m, score=m["auc"])
    if kind == "vertex":
        m = vertex_resolution(truth["vertex"], pred, truth["direction"])
        return dict(m, score=-m["q68_cm"])
    if kind == "direction":
        m = angular_resolution(truth["direction"], pred)
        return dict(m, score=-m["q68_deg"])
    if kind == "momentum":
        m = energy_resolution(truth["momentum"], 1000.0 * np.exp(pred))
        return dict(m, score=-m["q68_pct"])
    raise ValueError(f"Unknown probe kind: {kind!r}")


def summary(kind, m):
    """One head's headline numbers."""
    if kind == "pid":
        return (
            f"AUC {m['auc']:.4f}  mis-ID @90% sig {m['bkg_misid@0.9sigeff']:.4f}  "
            f"@99% sig {m['bkg_misid@0.99sigeff']:.4f}"
        )
    if kind == "vertex":
        return f"q68 {m['q68_cm']:.1f} cm  long {m['long_q68_cm']:.1f} cm  trans {m['trans_q68_cm']:.1f} cm"
    if kind == "direction":
        return f"q68 {m['q68_deg']:.2f} deg  median {m['median_deg']:.2f} deg"
    if kind == "momentum":
        return f"|dp/p| q68 {m['q68_pct']:.2f}%  sigma {m['res_pct']:.2f}%  bias {m['bias_pct']:+.2f}%"
    raise ValueError(f"Unknown probe kind: {kind!r}")


@HOOKS.register_module()
class WANDProbeEvaluator(HookBase):
    """Scores every head of a WANDEventProbes model on the val probe events (each head on its own task's events) every
    ``every_n_steps`` steps when ``cfg.evaluate`` is true, and selects each head on its own: at a new best score
    (head_metrics) its weights go to ``model/best_<head>.pth`` (``state_dict``, ``step``, ``score``, ``metrics``) and
    ``model/best_heads.json`` holds every head's best. Logs ``val/<head>/<metric>``; the checkpoint saver's
    ``selection_head`` selects one head for ``val/score``; None uses their mean.
    The validation loader is read whole on one GPU.
    """

    def __init__(self, every_n_steps, tasks, selection_head=None):
        self.every_n_steps, self.tasks, self.best = (
            int(every_n_steps),
            [dict(t) for t in tasks],
            {},
        )
        self.selection_head = selection_head

    def before_train(self):
        assert (
            comm.get_world_size() == 1
        ), "WANDProbeEvaluator scores the whole val set on one GPU"

    def after_step(self):
        info = self.trainer.comm_info
        step = info["epoch"] * info["iter_per_epoch"] + info["iter"] + 1
        if self.trainer.cfg.evaluate and step % self.every_n_steps == 0:
            self.eval(step)

    def eval(self, step):
        model = self.trainer.model
        model.eval()
        preds, keys = {}, (
            "task_mask",
            "probe_label",
            "vertex",
            "direction",
            "momentum",
        )
        truth = {k: [] for k in keys}
        with torch.no_grad():
            for input_dict in self.trainer.val_loader:
                input_dict = {
                    k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in input_dict.items()
                }
                for head, pred in model(input_dict)["probe_pred"].items():
                    preds.setdefault(head, []).append(pred.cpu())
                for k in keys:
                    truth[k].append(input_dict[k].cpu())
        truth = {k: torch.cat(v).numpy() for k, v in truth.items()}
        model_dir = os.path.join(self.trainer.cfg.save_path, "model")
        state = {k.removeprefix("module."): v for k, v in model.state_dict().items()}
        names, scores = [task["name"] for task in self.tasks], {}
        for head, pred in preds.items():
            t = names.index(head.split("_", 1)[1])  # heads are <probe>_<task>
            kind, mask = self.tasks[t]["kind"], truth["task_mask"][:, t]
            m = head_metrics(
                kind,
                torch.cat(pred).numpy()[mask],
                {k: v[mask] for k, v in truth.items()},
                truth["probe_label"][mask, t],
            )
            path = os.path.join(model_dir, f"best_{head}.pth")
            if head not in self.best:  # a resumed run continues from the saved best
                self.best[head] = (
                    torch.load(path, map_location="cpu")
                    if os.path.exists(path)
                    else dict(score=-np.inf)
                )
            scores[head], new = m["score"], m["score"] > self.best[head]["score"]
            if new:
                weights = {
                    k: v.cpu()
                    for k, v in state.items()
                    if k.startswith(f"heads.{head}.")
                }
                self.best[head] = dict(
                    state_dict=weights, step=step, score=m["score"], metrics=m
                )
                torch.save(self.best[head], path + ".tmp")
                os.replace(path + ".tmp", path)
            self.trainer.logger.info(
                f"val {head:22s} n {m['n']:5d}  {summary(kind, m)}"
                f"{'  (best)' if new else ''}"
            )
            if self.trainer.writer is not None:
                for name, value in m.items():
                    self.trainer.writer.add_scalar(f"val/{head}/{name}", value, step)
        with open(os.path.join(model_dir, "best_heads.json"), "w") as f:
            json.dump(
                {
                    h: dict(step=b["step"], metrics=b["metrics"])
                    for h, b in self.best.items()
                },
                f,
                indent=1,
            )
        headline = (
            scores[self.selection_head]
            if self.selection_head is not None
            else np.mean(list(scores.values()))
        )
        self.trainer.comm_info["current_metric_value"] = headline
        self.trainer.comm_info["current_metric_name"] = "val/score"
        model.train()

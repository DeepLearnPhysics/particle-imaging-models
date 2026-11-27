"""
Held-out event-classification evaluator for the supervised EventClassifier.

Runs the model on the val (held-out) loader, argmaxes ``cls_logits`` per event,
and reports accuracy + per-class precision/recall/F1 + mF1 + confusion matrix.
"""

from __future__ import annotations

import numpy as np
import torch

import pimm.utils.comm as comm
from pimm.distributed import unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase, get_writer_step
from pimm.observability import structured_logger as sl


@HOOKS.register_module()
class EventClassificationEvaluator(HookBase):
    """Evaluate the model's own event predictions on held-out data.

    For the supervised WC ``EventClassifier``: on rank 0, runs the model over
    ``trainer.val_loader`` (the held-out split), argmaxes ``cls_logits`` per
    event, and reports overall accuracy, per-class precision/recall/F1, macro F1
    (``mF1``), and a confusion matrix. Per-point labels are reduced to one label
    per event using ``offset``. It publishes ``mF1`` as the checkpoint-selection
    metric (``current_metric_value`` = ``mf1``, ``current_metric_name`` =
    ``<prefix>/mF1``). Runs feature extraction in fp32. Runs
    after every step when ``every_n_steps > 0`` (when
    ``(global_iter + 1) % every_n_steps == 0``), otherwise after each epoch; only
    when ``cfg.evaluate`` is true and ``val_loader`` is not ``None``. Registered
    as ``EventClassificationEvaluator`` (use as ``type`` in a ``hooks=[...]``
    entry).

    Args:
        every_n_steps (int): Step cadence; ``0`` evaluates once per epoch.
            Defaults to ``0``.
        label_key (str): Input key holding the per-event (or per-point) class
            label. Defaults to ``"event_label"``.
        prefix (str): Writer/metric prefix (slashes stripped). Defaults to
            ``"event_cls"``.
        class_names (list | None): Class names for logging; falls back to
            ``cfg.data.names``. Defaults to ``None``.
        metric (str): Selection metric: ``"mF1"``, or for two classes
            ``"auroc"`` / ``"eff_at_1pct"`` (class 1's efficiency at 1% of
            class 0 accepted), both from class 1's softmax score.
            Defaults to ``"mF1"``.

    Note:
        Runs on rank 0 only (the unwrapped model is used to avoid DDP
        collectives). The selection metric is macro F1 (``mF1``); higher is
        better. Unlike the probe evaluators, this scores the model's own
        classification head — no linear probe is fit.

    Example:
        Add to ``cfg.hooks`` for the supervised WC EventClassifier; it scores
        the model's own ``cls_logits`` on the held-out split:

        .. code-block:: python

            hooks = [
                dict(type="EventClassificationEvaluator", every_n_steps=1000,
                     label_key="event_label"),
            ]
    """

    def __init__(
        self,
        every_n_steps=0,
        label_key="event_label",
        prefix="event_cls",
        class_names=None,
        metric="mF1",
    ):
        self.metric = metric
        self.every_n_steps = int(every_n_steps)
        self.label_key = label_key
        self.prefix = str(prefix).strip("/")
        self.class_names = class_names

    def after_step(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps > 0:
            ci = self.trainer.comm_info
            g = ci["iter"] + ci["iter_per_epoch"] * ci["epoch"]
            if (g + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps == 0:
            self.eval()

    @torch.no_grad()
    @sl.log_trace_span("evaluation.event_classification")
    def eval(self):
        sl.add_step_tag("evaluation")
        if comm.get_rank() != 0:
            if comm.get_world_size() > 1:
                comm.synchronize()
            return

        self.trainer.logger.info(
            ">>>>>>>> Event Classification Eval (held-out) >>>>>>>>"
        )
        self.trainer.model.eval()
        model = unwrap_model(
            self.trainer.model
        )  # avoid DDP collectives in single-rank eval
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        preds, scores, labels = [], [], []
        for input_dict in self.trainer.val_loader:
            for k, v in list(input_dict.items()):
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device, non_blocking=True)
            # Full-event sparse-convolution evaluation needs fp32 for supported kernels.
            with torch.amp.autocast(device_type="cuda", enabled=False):
                out = model(input_dict)
            logits = out["cls_logits"].float()
            n = logits.shape[0]
            lab = input_dict[self.label_key].detach().long().view(-1)
            if lab.numel() != n:  # per-point -> per-event
                offs = [0] + input_dict["offset"].detach().cpu().tolist()
                lab = torch.stack([lab[s] for s, _ in zip(offs[:-1], offs[1:])])
            preds.append(logits.argmax(dim=-1).cpu())
            scores.append(
                (logits[:, -1] - logits[:, 0]).cpu()
            )  # logit margin: softmax saturates at 1.0 into ties
            labels.append(lab.cpu())

        y_pred = torch.cat(preds).numpy()
        y_true = torch.cat(labels).numpy()
        if y_true.size != len(
            self.trainer.val_loader.dataset
        ):  # a sharded (multi-GPU) val loader: rank 0's part only
            self.trainer.logger.warning(
                f"Event cls scored {y_true.size} of {len(self.trainer.val_loader.dataset)} val events"
            )
        names = self.class_names or getattr(self.trainer.cfg.data, "names", None)
        num_classes = max(
            int(y_true.max()) + 1 if y_true.size else 1,
            len(names) if names else 0,
        )

        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        acc = float((y_pred == y_true).mean()) if y_true.size else 0.0
        f1s, precs, recs = [], [], []
        for c in range(num_classes):
            tp = cm[c, c]
            prec = tp / max(cm[:, c].sum(), 1)
            rec = tp / max(cm[c, :].sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
        mf1 = float(np.mean(f1s))
        metrics = dict(mF1=mf1)
        if num_classes == 2 and 0 < y_true.sum() < y_true.size:
            from sklearn.metrics import roc_auc_score, roc_curve

            score = torch.cat(scores).numpy()
            fpr, tpr, _ = roc_curve(y_true, score)
            metrics.update(
                auroc=float(roc_auc_score(y_true, score)),
                eff_at_1pct=float(tpr[fpr <= 0.01].max()),
            )
            self.trainer.logger.info(
                f"Event cls (held-out): AUROC={metrics['auroc']:.4f} eff@1%={metrics['eff_at_1pct']:.4f}"
            )

        self.trainer.logger.info(
            f"Event cls (held-out): acc={acc:.4f} mF1={mf1:.4f} " f"n={y_true.size}"
        )
        for c in range(num_classes):
            nm = names[c] if names and c < len(names) else str(c)
            self.trainer.logger.info(
                f"  class {c} ({nm}): P={precs[c]:.4f} R={recs[c]:.4f} "
                f"F1={f1s[c]:.4f} support={int(cm[c].sum())}"
            )
        self.trainer.logger.info(
            "confusion (rows=true cols=pred):\n" + np.array2string(cm)
        )

        if self.trainer.writer is not None:
            step = get_writer_step(self.trainer)
            pf = f"{self.prefix}/" if self.prefix else ""
            self.trainer.writer.add_scalar(f"{pf}val/accuracy", acc, step)
            for k, v in metrics.items():
                self.trainer.writer.add_scalar(f"{pf}val/{k}", v, step)
            for c in range(num_classes):
                nm = names[c] if names and c < len(names) else str(c)
                self.trainer.writer.add_scalar(f"{pf}val/cls_{c}-{nm} F1", f1s[c], step)

        self.trainer.comm_info["current_metric_name"] = f"{self.prefix}/{self.metric}"
        self.trainer.comm_info["current_metric_value"] = metrics[self.metric]
        self.trainer.model.train()
        if comm.get_world_size() > 1:
            comm.synchronize()
        self.trainer.logger.info("<<<<<<<< End Event Classification Eval <<<<<<<<")

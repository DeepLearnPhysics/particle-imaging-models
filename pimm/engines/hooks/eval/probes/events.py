"""Event-feature linear probes with optional nested cross-validation."""

from __future__ import annotations

from datetime import timedelta
from functools import wraps
from weakref import WeakKeyDictionary, ref
import torch.distributed as dist
import os
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import pimm.utils.comm as comm
from pimm.distributed import unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase, get_writer_step, unsharded_val_loader
from pimm.observability import structured_logger as sl

_evaluation_groups = WeakKeyDictionary()


def _evaluation_group(training_group):
    """Reuse the evaluation group without keeping destroyed process groups alive."""
    cached = _evaluation_groups.get(training_group)
    group = cached() if cached is not None else None
    if group is None:
        group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
        _evaluation_groups[training_group] = ref(group)
    return group


def rank_zero_evaluation(evaluate):
    """Run an evaluator on rank zero and communicate completion on CPU.

    Every rank must enter this decorator in the same order. The wrapped method
    must not perform distributed collectives: peers wait on a separate Gloo
    group while rank zero runs it. No training-group collective is outstanding
    during evaluation, so its shorter NCCL timeout does not limit evaluation.
    Python exceptions are reported to peers before being reraised on rank zero,
    and the model's original training mode is restored even on failure.
    """

    @wraps(evaluate)
    def wrapped(self, *args, **kwargs):
        distributed = dist.is_available() and dist.is_initialized()
        distributed = distributed and dist.get_world_size() > 1
        group = _evaluation_group(dist.group.WORLD) if distributed else None
        rank = dist.get_rank() if distributed else 0
        status = [None]
        result = None
        error = None
        traceback = None
        if rank == 0:
            model = self.trainer.model
            was_training = model.training
            try:
                try:
                    result = evaluate(self, *args, **kwargs)
                finally:
                    model.train(was_training)
            except BaseException as exc:
                error = exc
                traceback = exc.__traceback__
                status[0] = (
                    f"Rank-zero evaluation {evaluate.__qualname__} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        if distributed:
            try:
                dist.broadcast_object_list(status, src=0, group=group)
            except BaseException:
                if error is not None:
                    raise error.with_traceback(traceback)
                raise
        if error is not None:
            raise error.with_traceback(traceback)
        if status[0] is not None:
            raise RuntimeError(status[0])
        return result

    return wrapped


OUTER_FOLDS = 5  # out-of-fold scoring over the task's events
INNER_FOLDS = 5  # hyperparameter CV over each outer fold's fit events
LOGREG_CS = tuple(
    10.0 ** np.arange(-4.0, 1.5, 0.5)
)  # logistic C grid, ascending (warm-started along it)
LOGREG_TOL = 1e-10  # Newton decrement (mean log loss) that ends a logistic fit, after one last full step
THREADS = 16  # CPU threads for the logistic fits (at most the process's CPUs)


# From scikit-learn's StratifiedKFold allocation (scikit-learn/scikit-learn).
def stratified_folds(y, folds, seed):
    """sklearn's StratifiedKFold(folds, shuffle=True, random_state=seed): each event's fold, every class dealt
    round robin over the folds (classes in order of appearance), then shuffled by one RandomState(seed).
    """
    rng = np.random.RandomState(seed)
    _, first, inverse = np.unique(y, return_index=True, return_inverse=True)
    label = np.unique(first, return_inverse=True)[1][inverse]
    allocation = np.stack(
        [
            np.bincount(np.sort(label)[i::folds], minlength=len(first))
            for i in range(folds)
        ]
    )
    fold = np.empty(len(y), np.int64)
    for k in range(len(first)):
        fold_of_class = np.repeat(np.arange(folds), allocation[:, k])
        rng.shuffle(fold_of_class)
        fold[label == k] = fold_of_class
    return fold


def auc(score, label):
    """ROC AUC as the rank statistic (ties count one half)."""
    _, inverse, count = np.unique(score, return_inverse=True, return_counts=True)
    rank = (np.cumsum(count) - (count - 1) / 2)[inverse]
    pos = label == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return (
        float((rank[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        if n1 and n0
        else float("nan")
    )


def _f1(pred, label, c):
    p, g = pred == c, label == c
    tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
    precision, recall = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
    f1, iou = 2 * precision * recall / (precision + recall + 1e-10), tp / (
        (p | g).sum() + 1e-10
    )
    return precision, recall, f1, iou


def classification_metrics(pred, score, label):
    """Macro IoU / precision / recall / F1 over the two classes (a class neither true nor predicted scores 0, except in
    IoU) and the AUC of the class-1 score."""
    per_class = np.array(
        [
            (
                _f1(pred, label, c)
                if (label == c).any() or (pred == c).any()
                else (0.0, 0.0, 0.0, 0.0)
            )
            for c in range(2)
        ],
        np.float64,
    )
    per_class[:, 3] = [
        _f1(pred, label, c)[3] for c in range(2)
    ]  # IoU counts every class
    m = per_class.mean(0)
    return dict(
        miou=float(m[3]),
        mprecision=float(m[0]),
        mrecall=float(m[1]),
        mf1=float(m[2]),
        auc=auc(score, label),
    )


def _log_loss(logit, y):
    """sklearn's log_loss of sigmoid(logit) against y in {0, 1}: probabilities clipped to [eps, 1 - eps]."""
    p = torch.sigmoid(logit)
    eps = torch.finfo(p.dtype).eps
    return float(-torch.where(y > 0, p, 1 - p).clamp(eps, 1 - eps).log().mean())


def _logistic_objective(a, y, ridge, w):
    """Summed log loss of the logits a @ w plus the L2 penalty sum(ridge w^2) / 2."""
    return float(
        F.binary_cross_entropy_with_logits(a @ w, y, reduction="sum")
        + 0.5 * (ridge * w * w).sum()
    )


def _logistic_fit(a, y, c, w, tol=LOGREG_TOL, hessian_every=4, max_iter=100):
    """sklearn's LogisticRegression(C=c) on a = [x, 1] (float64), run to convergence from w: argmin sum log loss +
    |w[:-1]|^2 / (2c), the intercept unpenalized. Newton steps with Armijo backtracking, the Hessian's Cholesky factor
    refreshed every `hessian_every` steps (Shamanskii's method); stops once a step's decrement per event is at most
    tol, after that last full step."""
    ridge = torch.full_like(w, 1.0 / c)
    ridge[-1] = 0.0
    loss = _logistic_objective(a, y, ridge, w)
    for i in range(max_iter):
        z = a @ w
        grad = a.T @ (torch.sigmoid(z) - y) + ridge * w
        if i % hessian_every == 0:
            hess = a.T @ (a * (torch.sigmoid(z) * torch.sigmoid(-z))[:, None])
            hess.diagonal().add_(ridge)
            factor = torch.linalg.cholesky(hess)
        step = torch.cholesky_solve(-grad[:, None], factor)[:, 0]
        slope = float(grad @ step)
        if -slope / len(a) <= tol:
            return w + step
        t, new = 1.0, _logistic_objective(a, y, ridge, w + step)
        while new > loss + 0.25 * t * slope:
            t /= 2
            new = _logistic_objective(a, y, ridge, w + t * step)
        w, loss = w + t * step, new
    return w


@contextmanager
def _cpu_threads(n):
    """torch's CPU threads set to n (at most the CPUs this process may use) inside, restored after."""
    before = torch.get_num_threads()
    torch.set_num_threads(min(n, len(os.sched_getaffinity(0))))
    try:
        yield
    finally:
        torch.set_num_threads(before)


def _standardized(x_fit, *xs):
    """[x, 1] design matrices (float64, CPU) of x_fit and xs, standardized by x_fit's mean and std."""
    x_fit = x_fit.double().cpu()
    mu, sigma = x_fit.mean(dim=0), x_fit.std(dim=0)
    sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
    return [
        torch.cat(
            [
                (x.double().cpu() - mu) / sigma,
                torch.ones(len(x), 1, dtype=torch.float64),
            ],
            1,
        )
        for x in (x_fit, *xs)
    ]


def logistic_cv(
    x_fit, y_fit, x_score, seed=0, folds=INNER_FOLDS, cs=LOGREG_CS, tol=LOGREG_TOL
):
    """One binary probe: sklearn's LogisticRegressionCV(Cs=cs, scoring='neg_log_loss',
    cv=StratifiedKFold(folds, shuffle=True, random_state=seed)) on the fit events, in torch. C is the lowest summed
    CV log loss (first on ties; each fold's path warm-started along the ascending cs), refit on all fit events from the
    folds' mean solution. Returns the class-1 probability of each x_score event and the chosen C.
    """
    a_fit, a_score = _standardized(x_fit, x_score)
    y = y_fit.double().cpu()
    fold = torch.from_numpy(stratified_folds(y_fit.cpu().numpy(), folds, seed))
    coefs = torch.zeros(folds, len(cs), a_fit.shape[1], dtype=torch.float64)
    cv_loss = np.zeros(len(cs))
    for f in range(folds):
        (a_in, y_in), (a_out, y_out) = (
            (a_fit[m], y[m]) for m in (fold != f, fold == f)
        )
        w = torch.zeros(a_fit.shape[1], dtype=torch.float64)
        for i, c in enumerate(cs):
            w = _logistic_fit(a_in, y_in, c, w, tol)
            coefs[f, i] = w
            cv_loss[i] += _log_loss(a_out @ w, y_out)
    best = int(
        np.argmin(cv_loss)
    )  # first minimum, as sklearn's argmax of the summed scores
    return (
        torch.sigmoid(
            a_score @ _logistic_fit(a_fit, y, cs[best], coefs[:, best].mean(dim=0), tol)
        ).numpy(),
        cs[best],
    )


def logistic_oof(x, y, seed=0, outer=OUTER_FOLDS, inner=INNER_FOLDS, threads=THREADS):
    """Binary PID, out of fold: x [N, C] event features, y [N] in {0, 1}. Returns metrics (classification_metrics plus
    best_C, the geometric mean of the folds' C) and dict(score, pred, true, fold) per event.
    """
    y_np = y.cpu().numpy().astype(np.int64)
    assert set(np.unique(y_np)) <= {0, 1}, "logistic_oof is binary"
    fold = stratified_folds(y_np, outer, seed)
    score, chosen = np.empty(len(y_np)), []
    with _cpu_threads(threads):
        for f in range(outer):
            fit, held = torch.from_numpy(fold != f), torch.from_numpy(fold == f)
            p, c = logistic_cv(x[fit], y[fit], x[held], seed + 1, inner)
            score[fold == f] = p
            chosen.append(c)
    pred = (score > 0.5).astype(np.int64)
    metrics = dict(
        classification_metrics(pred, score, y_np),
        best_C=float(np.exp(np.mean(np.log(chosen)))),
    )
    return metrics, dict(score=score, pred=pred, true=y_np, fold=fold)


@HOOKS.register_module()
class EventLinearProbeEvaluator(HookBase):
    """Probe frozen event features with the public train/test split by default.

    out_of_fold=True enables nested binary logistic cross-validation; seeds and
    feature_sampling configure that mode.
    """

    def __init__(
        self,
        label_key="event_label",
        every_n_steps=0,
        train_fraction=0.5,
        max_events_per_class=None,
        seed=0,
        prefix="event_probe",
        class_names=None,
        train_config=None,
        write_cls_metrics=True,
        require_heldout_data=True,
        out_of_fold=False,
        seeds=(0, 1, 2),
        feature_sampling=None,
    ):
        self.label_key = label_key
        self.every_n_steps = int(every_n_steps)
        self.train_fraction = float(train_fraction)
        self.max_events_per_class = max_events_per_class
        self.seed = int(seed)
        self.prefix = str(prefix).strip("/")
        self.class_names = class_names
        self.train_config = train_config
        self.write_cls_metrics = bool(write_cls_metrics)
        self.require_heldout_data = bool(require_heldout_data)
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")

        self.out_of_fold = bool(out_of_fold)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.feature_sampling = feature_sampling
        self._validated = False
        if self.out_of_fold:
            if not self.seeds:
                raise ValueError("seeds must name at least one fold seed")
            if class_names is not None:
                self._check_binary(class_names)
        elif feature_sampling is not None:
            raise ValueError("feature_sampling requires out_of_fold=True")

    def before_train(self):
        if self.out_of_fold:
            class_names = self._resolved_class_names()
            if class_names is not None:
                self._check_binary(class_names)

    def after_step(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"]
                * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps == 0:
            self.eval()

    def _move_to_device(self, input_dict):
        """Move tensor batch entries to the evaluator device in place."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device, non_blocking=True)
        return input_dict

    def _forward_point(self, input_dict):
        """Run the model and return the Point object required by the probe."""
        model = unwrap_model(self.trainer.model)
        use_amp = bool(getattr(self.trainer.cfg, "enable_amp", False))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_dtype = getattr(self.trainer.cfg, "amp_dtype", "bfloat16")
        dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

        if self.out_of_fold:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                output = model(input_dict, return_point=True)
        else:
            with torch.inference_mode():
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        output = model(input_dict, return_point=True)
                else:
                    output = model(input_dict, return_point=True)

        if isinstance(output, dict) and "point" in output:
            point = output["point"]
            if (
                self.out_of_fold
                and self.feature_sampling is not None
                and "pooling_parent" in point.keys()
            ):
                point = point.upcast(feature_sampling=self.feature_sampling)
            return point
        raise KeyError(
            "EventLinearProbeEvaluator expected model(..., return_point=True) "
            "to return a dict containing 'point'."
        )

    @staticmethod
    def _labels_by_event(labels, offsets, n_points):
        labels = labels.detach().long().view(-1)
        n_events = len(offsets) - 1
        if labels.numel() == n_events:
            return labels.cpu()
        if labels.numel() == n_points:
            event_labels = []
            for start, end in zip(offsets[:-1], offsets[1:]):
                event_labels.append(labels[start:end][0])
            return torch.stack(event_labels).cpu()
        raise ValueError(
            f"Cannot map {labels.numel()} labels to {n_events} events "
            f"and {n_points} points."
        )

    @staticmethod
    def _loader_dataset(loader):
        return getattr(loader, "dataset", None)

    @classmethod
    def _base_dataset(cls, dataset):
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return dataset

    @classmethod
    def _dataset_split(cls, dataset):
        dataset = cls._base_dataset(dataset)
        split = getattr(dataset, "split", None)
        if split is None:
            return None
        if isinstance(split, (list, tuple, set)):
            return tuple(str(item) for item in split)
        return str(split)

    @staticmethod
    def _source_key(source, source_idx):
        if isinstance(source, dict):
            source_root = source.get("source_root") or source.get("data_root")
            if source_root is not None:
                return os.path.realpath(str(source_root))
            if source.get("name") is not None:
                return str(source["name"])
        return str(source_idx)

    @classmethod
    def _event_keys(cls, dataset):
        if dataset is None:
            return None

        subset_indices = None
        while isinstance(dataset, Subset):
            indices = [int(idx) for idx in dataset.indices]
            if subset_indices is None:
                subset_indices = indices
            else:
                subset_indices = [indices[idx] for idx in subset_indices]
            dataset = dataset.dataset

        data_list = getattr(dataset, "data_list", None)
        sources = getattr(dataset, "datasets", None)
        if data_list is None or sources is None or len(data_list) == 0:
            return None

        if subset_indices is None:
            indices = range(len(data_list))
        else:
            indices = subset_indices

        keys = set()
        base_len = len(data_list)
        for dataset_idx in indices:
            source_idx, event_idx = data_list[int(dataset_idx) % base_len]
            source_key = cls._source_key(sources[int(source_idx)], int(source_idx))
            keys.add((source_key, int(event_idx)))
        return keys

    @staticmethod
    def _format_event_key(key):
        source_key, event_idx = key
        return f"{source_key}:{event_idx}"

    def _validate_heldout_source(self):
        val_dataset = self._loader_dataset(self.trainer.val_loader)
        val_split = self._dataset_split(val_dataset)
        heldout_splits = {"holdout", "val", "test"}
        forbidden_splits = {"train", "all"}

        if val_split is None:
            raise RuntimeError(
                "EventLinearProbeEvaluator cannot verify heldout evaluation: "
                "val_loader.dataset has no split metadata."
            )

        val_split_set = {val_split} if isinstance(val_split, str) else set(val_split)
        if val_split_set & forbidden_splits or not val_split_set <= heldout_splits:
            raise RuntimeError(
                "EventLinearProbeEvaluator refuses to run on non-heldout data: "
                f"val_loader split={val_split!r}. Expected one of "
                f"{sorted(heldout_splits)}."
            )

        train_loader = getattr(self.trainer, "train_loader", None)
        train_dataset = self._loader_dataset(train_loader)
        train_keys = self._event_keys(train_dataset)
        val_keys = self._event_keys(val_dataset)
        if train_keys is None or val_keys is None:
            raise RuntimeError(
                "EventLinearProbeEvaluator cannot verify train/heldout "
                "disjointness for these datasets. Use a dataset exposing "
                "`data_list` and `datasets`, or disable require_heldout_data "
                "only for a deliberate diagnostic run."
            )

        overlap = train_keys & val_keys
        if overlap:
            examples = ", ".join(
                self._format_event_key(key) for key in sorted(overlap)[:5]
            )
            raise RuntimeError(
                "EventLinearProbeEvaluator detected train/heldout leakage: "
                f"{len(overlap)} events appear in both loaders. Examples: "
                f"{examples}"
            )

        self.trainer.logger.info(
            "Event probe data guard: val split=%s, train events=%d, "
            "heldout events=%d, overlap=0",
            val_split,
            len(train_keys),
            len(val_keys),
        )

    def _process_batch(self, input_dict):
        input_dict = self._move_to_device(input_dict)
        point = self._forward_point(input_dict)

        features = point.feat.float()
        point_offset = (
            point["offset"] if "offset" in point.keys() else input_dict["offset"]
        )
        offsets = [0] + point_offset.detach().cpu().tolist()
        labels = self._labels_by_event(
            input_dict[self.label_key], offsets, features.shape[0]
        )

        embeddings = []
        kept_labels = []
        for event_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end <= start:
                continue
            embeddings.append(features[start:end].mean(dim=0).cpu())
            kept_labels.append(labels[event_idx])
        return embeddings, kept_labels

    def _collect_features(self):
        embeddings = []
        labels = []
        loader = self._eval_loader()
        for batch_idx, input_dict in enumerate(loader):
            batch_embeddings, batch_labels = self._process_batch(input_dict)
            embeddings.extend(batch_embeddings)
            labels.extend(batch_labels)
            if (batch_idx + 1) % 10 == 0:
                self.trainer.logger.info(
                    "Event probe feature extraction: "
                    f"{batch_idx + 1}/{len(loader)} batches"
                )

        if not embeddings:
            raise RuntimeError("No event embeddings were collected from val_loader")
        X = torch.stack(embeddings, dim=0)
        y = torch.stack(labels, dim=0).long()
        return X, y

    def _split_train_test(self, X, y):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_indices = []
        test_indices = []
        classes = sorted(int(c) for c in y.unique().tolist())
        if len(classes) < 2:
            raise RuntimeError(
                "Event probe requires at least two classes in the holdout set; "
                f"got {classes}"
            )

        for class_id in classes:
            idx = torch.nonzero(y == class_id, as_tuple=False).flatten()
            idx = idx[torch.randperm(idx.numel(), generator=generator)]
            if self.max_events_per_class is not None:
                idx = idx[: int(self.max_events_per_class)]
            if idx.numel() < 2:
                raise RuntimeError(
                    f"Need at least two holdout events for class {class_id}, "
                    f"got {idx.numel()}"
                )
            n_train = int(round(idx.numel() * self.train_fraction))
            n_train = max(1, min(n_train, idx.numel() - 1))
            train_indices.append(idx[:n_train])
            test_indices.append(idx[n_train:])

        train_idx = torch.cat(train_indices, dim=0)
        test_idx = torch.cat(test_indices, dim=0)
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    @sl.log_trace_span("evaluation.event_linear_probe")
    def eval(self):
        if self.out_of_fold:
            return self._eval_oof()
        sl.add_step_tag("evaluation")
        if comm.get_rank() != 0:
            if comm.get_world_size() > 1:
                comm.synchronize()
            return

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Event Linear Probe Evaluation >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()

        if self.require_heldout_data:
            self._validate_heldout_source()

        X, y = self._collect_features()
        X_train, y_train, X_test, y_test = self._split_train_test(X, y)
        self.trainer.logger.info(
            "Event probe heldout split: "
            f"probe_train={tuple(X_train.shape)}, "
            f"heldout_val={tuple(X_test.shape)}"
        )

        self._evaluate_probe(X_train, y_train, X_test, y_test)

        self.trainer.model.train()
        if comm.get_world_size() > 1:
            comm.synchronize()

    def _evaluate_probe(self, X_train, y_train, X_test, y_test):
        from .points import LinearProbingTrainer

        class_names = self.class_names
        if class_names is None:
            class_names = getattr(self.trainer.cfg.data, "names", None)
        if class_names is None:
            num_classes = int(max(y_train.max(), y_test.max()).item()) + 1
            class_names = [str(i) for i in range(num_classes)]
        else:
            num_classes = max(
                len(class_names),
                int(max(y_train.max(), y_test.max()).item()) + 1,
            )

        probe = LinearProbingTrainer(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            logger=self.trainer.logger,
            config=self.train_config,
        )
        metrics = probe.train_and_evaluate()

        m_iou = metrics["m_iou"]
        m_precision = metrics["m_precision"]
        m_recall = metrics["m_recall"]
        m_f1 = metrics["m_f1"]
        iou_class = metrics["iou_class"]
        precision_class = metrics["precision_class"]
        recall_class = metrics["recall_class"]
        f1_class = metrics["f1_class"]
        cm = metrics["confusion_matrix"]
        support = metrics["class_support"]

        self.trainer.logger.info(
            "Event probe result: mIoU/mPrec/mRec/mF1 "
            f"{m_iou:.4f}/{m_precision:.4f}/{m_recall:.4f}/{m_f1:.4f}."
        )
        for class_id in range(num_classes):
            name = (
                class_names[class_id] if class_id < len(class_names) else str(class_id)
            )
            self.trainer.logger.info(
                "Event probe class "
                f"{class_id} ({name}): IoU={iou_class[class_id]:.4f}, "
                f"Precision={precision_class[class_id]:.4f}, "
                f"Recall={recall_class[class_id]:.4f}, "
                f"F1={f1_class[class_id]:.4f}, Support={support[class_id]}"
            )
        self.trainer.logger.info(
            "Event probe confusion matrix rows=true cols=pred:\n" + np.array2string(cm)
        )

        if self.trainer.writer is not None:
            step = get_writer_step(self.trainer)
            prefix = f"{self.prefix}/" if self.prefix else ""
            self.trainer.writer.add_scalar(f"{prefix}val/mIoU", m_iou, step)
            self.trainer.writer.add_scalar(f"{prefix}val/mPrecision", m_precision, step)
            self.trainer.writer.add_scalar(f"{prefix}val/mRecall", m_recall, step)
            self.trainer.writer.add_scalar(f"{prefix}val/mF1", m_f1, step)
            if self.write_cls_metrics:
                for class_id in range(num_classes):
                    name = (
                        class_names[class_id]
                        if class_id < len(class_names)
                        else str(class_id)
                    )
                    self.trainer.writer.add_scalar(
                        f"{prefix}val/cls_{class_id}-{name} IoU",
                        iou_class[class_id],
                        step,
                    )
                    self.trainer.writer.add_scalar(
                        f"{prefix}val/cls_{class_id}-{name} F1",
                        f1_class[class_id],
                        step,
                    )

        metric_name = f"{self.prefix}/mF1" if self.prefix else "mF1"
        self.trainer.comm_info["current_metric_name"] = metric_name
        self.trainer.comm_info["current_metric_value"] = m_f1
        self.trainer.comm_info[f"{self.prefix}_current_metric_value"] = m_f1
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<<< End Event Linear Probe Evaluation <<<<<<<<<<<<<<<<<"
        )

    @staticmethod
    def _check_binary(class_names):
        if len(class_names) != 2:
            raise TypeError(
                "EventLinearProbeEvaluator is binary (logistic_oof); "
                f"got {len(class_names)} classes {list(class_names)}"
            )

    def _resolved_class_names(self):
        data = getattr(self.trainer.cfg, "data", None)
        return self.class_names or getattr(data, "names", None)

    def _eval_loader(self):
        """The whole held-out set, on rank zero without distributed sharding."""
        return unsharded_val_loader(self.trainer)

    @sl.log_trace_span("evaluation.event_linear_probe")
    @rank_zero_evaluation
    def _eval_oof(self):
        sl.add_step_tag("evaluation")

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Event Linear Probe Evaluation >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()

        if self.require_heldout_data and not self._validated:
            self._validate_heldout_source()
            self._validated = True

        X, y = self._collect_features()
        self.trainer.logger.info(
            f"Event probe: {tuple(X.shape)} heldout features scored out of fold, "
            f"seeds {list(self.seeds)}"
        )

        self._evaluate_oof_probe(X, y)

        self.trainer.model.train()

    def _evaluate_oof_probe(self, X, y):
        class_names = self._resolved_class_names() or ["0", "1"]
        self._check_binary(class_names)
        runs, matrices = [], []
        for seed in self.seeds:
            metrics, out = logistic_oof(X, y, seed)
            per_class, cm = _class_scores(out["pred"], out["true"], class_names)
            runs.append(dict(metrics, **per_class))
            matrices.append(cm)
        metrics = _mean_over_seeds(runs)

        self.trainer.logger.info(
            "Event probe result (out of fold): mIoU/mPrec/mRec/mF1/AUC "
            f"{metrics['miou']:.4f}/{metrics['mprecision']:.4f}/{metrics['mrecall']:.4f}/"
            f"{metrics['mf1']:.4f}/{metrics['auc']:.4f}, C={metrics['best_C']:.3g}."
        )
        for class_id, name in enumerate(class_names):
            self.trainer.logger.info(
                f"Event probe class {class_id} ({name}): "
                f"IoU={metrics[f'cls_{class_id}-{name} IoU']:.4f}, "
                f"F1={metrics[f'cls_{class_id}-{name} F1']:.4f}, "
                f"Support={int(matrices[0][class_id].sum())}"
            )
        self.trainer.logger.info(
            f"Event probe confusion matrix (seed {self.seeds[0]}) rows=true cols=pred:\n"
            + np.array2string(matrices[0])
        )

        if self.trainer.writer is not None:
            step = get_writer_step(self.trainer)
            prefix = f"{self.prefix}/" if self.prefix else ""
            for key, value in metrics.items():
                if self.write_cls_metrics or not key.startswith("cls_"):
                    self.trainer.writer.add_scalar(f"{prefix}oof/{key}", value, step)

        metric_name = f"{self.prefix}/oof/mf1" if self.prefix else "oof/mf1"
        self.trainer.comm_info["current_metric_name"] = metric_name
        self.trainer.comm_info["current_metric_value"] = metrics["mf1"]
        self.trainer.comm_info[f"{self.prefix}_current_metric_value"] = metrics["mf1"]
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<<< End Event Linear Probe Evaluation <<<<<<<<<<<<<<<<<"
        )


def _mean_over_seeds(runs):
    """Each metric's mean over the seeds' runs, plus ``<metric>_std`` when there are several."""
    out = {}
    for key in runs[0]:
        values = np.array([run[key] for run in runs], dtype=np.float64)
        out[key] = float(values.mean())
        if len(runs) > 1:
            out[f"{key}_std"] = float(values.std())
    return out


def _class_scores(pred, true, class_names):
    """Per-class IoU and F1 of binary out-of-fold predictions (keys ``cls_<id>-<name> IoU`` / ``F1``) and the
    confusion matrix, rows true and columns predicted."""
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (np.asarray(true), np.asarray(pred)), 1)
    tp = np.diag(cm).astype(np.float64)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-10)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-10)
    scores = {}
    for class_id, name in enumerate(class_names):
        scores[f"cls_{class_id}-{name} IoU"] = float(iou[class_id])
        scores[f"cls_{class_id}-{name} F1"] = float(f1[class_id])
    return scores, cm

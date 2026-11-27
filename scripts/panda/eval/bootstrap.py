"""Bootstrap images: python scripts/panda/eval/bootstrap.py INPUT --column ari --metric mean --out ci.json.

WC uses PREFIX --metric wc; panoptic uses EVENTS.npz --metric panoptic --num-classes N --noise-class K.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def percentile(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _resample_weights(n, rng, count, strata=None):
    if n < 1 or count < 1:
        raise ValueError("The input and resample count must be nonempty.")
    groups = None
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape != (n,):
            raise ValueError("Strata must contain one label per image.")
        groups = [np.flatnonzero(strata == label) for label in np.unique(strata)[::-1]]
    for start in range(0, count, 50):
        weights = []
        for _ in range(min(50, count - start)):
            if groups is None:
                draw = rng.integers(0, n, n)
            else:
                draw = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
            weights.append(np.bincount(draw, minlength=n))
        yield np.stack(weights).astype(np.float64)


def boot_mean(x, seed=0, resamples=1000, strata=None):
    x = np.asarray(x, np.float64)
    if x.ndim != 1:
        raise ValueError("Mean bootstrap expects one value per image.")
    rng = np.random.default_rng(seed)
    samples = np.concatenate([w @ x / len(x) for w in _resample_weights(len(x), rng, resamples, strata)])
    return percentile(samples)


def boot_cm(cms, metric, seed=0, resamples=1000, strata=None):
    cms = np.asarray(cms)
    if cms.ndim != 3:
        raise ValueError("Confusion bootstrap expects (images, true classes, predicted classes).")
    n, rows, cols = cms.shape
    flat = cms.reshape(n, -1).astype(np.float64)
    rng = np.random.default_rng(seed)
    samples = []
    for weights in _resample_weights(n, rng, resamples, strata):
        for row in weights @ flat:
            samples.append(metric(row.reshape(rows, cols)))
    return percentile(samples)


def prf_macro(cm):
    cm = np.asarray(cm, np.float64)
    n = len(cm)
    tp, true, pred = np.diag(cm), cm.sum(1), cm.sum(0)[:n]
    p = np.divide(tp, pred, out=np.zeros(n), where=pred > 0)
    r = np.divide(tp, true, out=np.zeros(n), where=true > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(n), where=(p + r) > 0)
    seen = (true > 0) | (pred > 0)
    return p[seen].mean(), r[seen].mean(), f[seen].mean()


def f1_of_macro(cm):
    p, r, _ = prf_macro(cm)
    return 2 * p * r / (p + r) if p + r else 0.0


def boot_quantile(x, seed=0, resamples=1000, quantile=0.68, strata=None):
    x = np.asarray(x, np.float64)
    if x.ndim != 1 or len(x) < 1 or resamples < 1:
        raise ValueError("Quantile bootstrap expects a nonempty image array and positive resample count.")
    rng = np.random.default_rng(seed)
    groups = [np.arange(len(x))]
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape != x.shape:
            raise ValueError("Strata must contain one label per image.")
        groups = [np.flatnonzero(strata == label) for label in np.unique(strata)[::-1]]
    samples = []
    for _ in range(resamples):
        draw = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        samples.append(np.quantile(x[draw], quantile))
    return percentile(samples)


def boot_auc(y, score, seed=0, resamples=1000):
    from sklearn.metrics import roc_auc_score

    y, score = np.asarray(y), np.asarray(score)
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg) or y.shape != score.shape:
        raise ValueError("AUC bootstrap requires scores from both binary classes.")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        a = pos[rng.integers(0, len(pos), len(pos))]
        b = neg[rng.integers(0, len(neg), len(neg))]
        samples.append(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
    return percentile(samples)


def _flatten_metrics(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(_flatten_metrics(item, name))
        elif isinstance(item, (int, float, np.number)):
            result[name] = float(item)
    return result


def _metric_intervals(point, samples):
    intervals = {}
    for key, values in samples.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        lo, hi = percentile(finite) if len(finite) else (float("nan"), float("nan"))
        intervals[key] = dict(value=point[key], lo=lo, hi=hi, finite_resamples=len(finite))
    return intervals


def _wc_groups(rows, rings, truths):
    events = {row["event"] for row in rows}
    if len(events) != len(rows):
        raise ValueError("WC rows must contain one record per image.")
    ring_groups, truth_groups = {e: [] for e in events}, {e: [] for e in events}
    for records, groups in ((rings, ring_groups), (truths, truth_groups)):
        for record in records:
            if record["event"] not in groups:
                raise ValueError("A WC particle record has no corresponding image row.")
            groups[record["event"]].append(record)
    return [(row, ring_groups[row["event"]], truth_groups[row["event"]]) for row in rows]


def _resample_wc_records(groups, draw):
    rows, rings, truths = [], [], []
    for event, index in enumerate(draw):
        row, event_rings, event_truths = groups[index]
        # Repeated images need distinct IDs for the particle-to-truth joins.
        rows.append(dict(row, event=event))
        rings.extend(dict(record, event=event) for record in event_rings)
        truths.extend(dict(record, event=event) for record in event_truths)
    return rows, rings, truths


def boot_wc(rows, rings, truths, seed=0, resamples=1000):
    if __package__:
        from .test_wc import summarize_records
    else:
        from test_wc import summarize_records

    if not rows or resamples < 1:
        raise ValueError("WC bootstrap requires image rows and a positive resample count.")
    groups = _wc_groups(rows, rings, truths)
    point = _flatten_metrics(summarize_records(rows, rings, truths)["michel_excluded"]["summary"])
    samples = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        records = _resample_wc_records(groups, rng.integers(0, len(rows), len(rows)))
        reduced = _flatten_metrics(summarize_records(*records)["michel_excluded"]["summary"])
        for key, values in samples.items():
            values.append(reduced.get(key, float("nan")))
    return dict(metric="wc", scope="michel_excluded.summary", events=len(rows), confidence=0.95,
                resamples=resamples, seed=seed, stratified=False, metrics=_metric_intervals(point, samples))


def boot_panoptic(data, num_classes, noise_class, seed=0, resamples=1000):
    from pimm.engines.metrics.panoptic import summarize_rows

    n = len(data["ari"])
    if n < 1 or resamples < 1 or num_classes < 1 or noise_class < 0:
        raise ValueError("Panoptic bootstrap needs images, positive classes/resamples, and a nonnegative noise class.")
    data = {key: value for key, value in data.items() if key not in ("semantic_cm", "noise_cm")}
    if any(len(values) != n for values in data.values()):
        raise ValueError("Panoptic arrays must contain one record per image.")
    rows = [{key: values[i] for key, values in data.items()} for i in range(n)]
    point = _flatten_metrics(summarize_rows(rows, num_classes, noise_class))
    samples = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        draw = rng.integers(0, n, n)
        reduced = _flatten_metrics(summarize_rows([rows[i] for i in draw], num_classes, noise_class))
        for key, values in samples.items():
            values.append(reduced.get(key, float("nan")))
    return dict(metric="panoptic", events=n, confidence=0.95, resamples=resamples, seed=seed,
                stratified=False, metrics=_metric_intervals(point, samples))


def load_wc(prefix):
    import pandas as pd

    prefix = str(prefix)
    rows = json.loads(Path(prefix + ".rows.json").read_text())
    rings = pd.read_parquet(prefix + ".rings.parquet").to_dict("records")
    truths = pd.read_parquet(prefix + ".truth.parquet").to_dict("records")
    return rows, rings, truths


def _load(path, column):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if column is None:
                if len(data.files) != 1:
                    raise ValueError("Use --column to select an array from an NPZ file.")
                column = data.files[0]
            return data[column]
    if path.suffix in (".parquet", ".csv"):
        import pandas as pd

        if column is None:
            raise ValueError("Use --column to select a table column.")
        data = pd.read_parquet(path, columns=[column]) if path.suffix == ".parquet" else pd.read_csv(path, usecols=[column])
        values = data[column].to_numpy()
        return np.stack(values) if values.dtype == object else values
    raise ValueError("Expected a .npy, .npz, .parquet, or .csv input.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--column")
    parser.add_argument("--metric", choices=("mean", "q68", "macro-f1", "f1-of-macro", "auc", "wc", "panoptic"), default="mean")
    parser.add_argument("--shape", type=int, nargs=2, metavar=("TRUE", "PRED"))
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--noise-class", type=int)
    parser.add_argument("--strata-column")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.metric != "panoptic" and (args.num_classes is not None or args.noise_class is not None):
        parser.error("num-classes and noise-class are only used by the panoptic metric.")
    if args.metric == "panoptic":
        if args.num_classes is None or args.noise_class is None:
            parser.error("Panoptic bootstrap requires --num-classes and --noise-class.")
        if args.column or args.strata_column or args.shape or args.scale != 1.0:
            parser.error("Panoptic takes an event NPZ and does not use column, strata, shape, or scale options.")
        with np.load(args.input, allow_pickle=False) as data:
            result = boot_panoptic(dict(data), args.num_classes, args.noise_class,
                                   seed=args.seed, resamples=args.resamples)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    if args.metric == "wc":
        if args.column or args.strata_column or args.shape or args.scale != 1.0:
            parser.error("WC takes an output prefix and does not use column, strata, shape, or scale options.")
        result = boot_wc(*load_wc(args.input), seed=args.seed, resamples=args.resamples)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    x = _load(args.input, args.column)
    strata = _load(args.input, args.strata_column) if args.strata_column else None
    options = dict(seed=args.seed, resamples=args.resamples)
    if args.metric == "auc":
        if strata is None:
            parser.error("AUC requires --strata-column with the binary labels.")
        from sklearn.metrics import roc_auc_score

        value = roc_auc_score(strata, x)
        lo, hi = boot_auc(strata, x, **options)
    elif args.metric in ("macro-f1", "f1-of-macro"):
        if args.shape:
            x = x.reshape(len(x), *args.shape)
        if x.ndim != 3 or x.shape[2] < x.shape[1]:
            parser.error("Confusion arrays need shape (images, true classes, predicted classes); use --shape for flattened matrices.")
        metric = (lambda cm: prf_macro(cm)[2]) if args.metric == "macro-f1" else f1_of_macro
        value = metric(x.sum(0, dtype=np.float64))
        lo, hi = boot_cm(x, metric, strata=strata, **options)
    elif args.metric == "q68":
        value = np.quantile(x, 0.68)
        lo, hi = boot_quantile(x, strata=strata, **options)
    else:
        value = np.mean(x)
        lo, hi = boot_mean(x, strata=strata, **options)
    result = dict(metric=args.metric, events=len(x), value=float(value * args.scale), lo=lo * args.scale,
                  hi=hi * args.scale, confidence=0.95, resamples=args.resamples, seed=args.seed,
                  stratified=strata is not None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""NEXT evaluation: python scripts/panda/eval/next.py {infer,features,probe,score} --help."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

B_ACCEPT = 0.01
MIN_BKG_FOM = 10


def _curve(scores, labels, w=None):
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    w = np.ones(len(s)) if w is None else w[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp, fp = np.cumsum(w * y)[last], np.cumsum(w * (1 - y))[last]
    return np.r_[0.0, tp], np.r_[0.0, fp]


def summary(scores, labels, w=None):
    tp, fp = _curve(scores, labels, w)
    tpr, fpr = tp / tp[-1], fp / fp[-1]
    eff = tpr[fpr <= B_ACCEPT + 1e-12].max()
    auroc = np.trapezoid(tpr, fpr)
    ok = fp >= MIN_BKG_FOM * (fp[-1] / max((labels == 0).sum(), 1))
    fom = (tpr[ok] / np.sqrt(fpr[ok])).max() if ok.any() else np.nan
    return dict(eff_at_1pct=eff, auroc=auroc, max_fom=fom)


def bootstrap(scores, labels, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    sig, bkg = labels == 1, labels == 0
    out = []
    for _ in range(n):
        w = np.zeros(len(scores))
        w[sig] = rng.multinomial(sig.sum(), np.full(sig.sum(), 1 / sig.sum()))
        w[bkg] = rng.multinomial(bkg.sum(), np.full(bkg.sum(), 1 / bkg.sum()))
        out.append(summary(scores, labels, w))
    return {k: np.array([o[k] for o in out]) for k in out[0]}


def report(draws, n=1000, seed=0):
    points = [summary(s, y) for s, y in draws]
    samples = [bootstrap(s, y, n, seed + i) for i, (s, y) in enumerate(draws)]
    res = {}
    for k in points[0]:
        pooled = np.concatenate([b[k] for b in samples])
        res[k] = dict(value=float(np.mean([p[k] for p in points])),
                      lo=float(np.nanpercentile(pooled, 2.5)), hi=float(np.nanpercentile(pooled, 97.5)))
    res["n_signal"] = int((draws[0][1] == 1).sum())
    res["n_background"] = int((draws[0][1] == 0).sum())
    return res


def val_threshold(s, y):
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp, fp = np.cumsum(yy)[last], np.cumsum(1 - yy)[last]
    fom = np.where(fp >= MIN_BKG_FOM, (tp / tp[-1]) / np.sqrt(np.maximum(fp, 1) / fp[-1]), -np.inf)
    i = int(np.argmax(fom))
    return float(ss[last[i]]), dict(eps_s=float(tp[i] / tp[-1]), eps_b=float(fp[i] / fp[-1]),
                                   fom=float(fom[i]), n_bkg_kept=int(fp[i]))


def at_threshold(s, y, t, n=1000, seed=0):
    keep, sig, bkg = s >= t, y == 1, y == 0
    ks, kb = keep[sig].astype(float), keep[bkg].astype(float)
    point = dict(eps_s=ks.mean(), eps_b=kb.mean())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        es = rng.multinomial(len(ks), np.full(len(ks), 1 / len(ks))) @ ks / len(ks)
        eb = rng.multinomial(len(kb), np.full(len(kb), 1 / len(kb))) @ kb / len(kb)
        draws.append((es, eb, es / np.sqrt(eb) if eb > 0 else np.nan))
    draws = np.array(draws)
    res = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        values = [*point.items(), ("fom", point["eps_s"] / np.sqrt(point["eps_b"]))]
    for j, (k, v) in enumerate(values):
        res[k] = dict(value=float(v), lo=float(np.nanpercentile(draws[:, j], 2.5)),
                      hi=float(np.nanpercentile(draws[:, j], 97.5)))
    res["n_bkg_kept"] = int(kb.sum())
    res["n_signal"], res["n_background"] = int(sig.sum()), int(bkg.sum())
    return res


def load_scores(path, key="score"):
    with np.load(path, allow_pickle=False) as d:
        s, y = d[key].astype(np.float64), d["label"].astype(np.int64)
    if s.ndim != 1 or y.shape != s.shape or set(np.unique(y)) != {0, 1} or not np.isfinite(s).all():
        raise ValueError("Scores must be finite [N] values with both binary classes in label.")
    return s, y


def score_files(test_paths, val_paths=None, n=1000, seed=0, key="score"):
    draws = [load_scores(p, key) for p in test_paths]
    result = report(draws, n=n, seed=seed)
    if val_paths:
        if len(val_paths) != len(test_paths):
            raise ValueError("Provide one validation score file per test draw.")
        fixed = []
        for i, (val, test) in enumerate(zip(val_paths, draws)):
            s, y = load_scores(val, key)
            if (y == 0).sum() < MIN_BKG_FOM:
                raise ValueError("Validation FOM selection requires at least 10 background events.")
            threshold, at_val = val_threshold(s, y)
            fixed.append(dict(threshold=threshold, val=at_val,
                              test=at_threshold(*test, threshold, n=n, seed=seed + i)))
        result["val_chosen_fom"] = fixed
    return result


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_dataset(config, split, data_root=None, cache_dir=None, max_events=None, workers=0, raw=False):
    from pimm.datasets import build_dataset

    spec = copy.deepcopy(dict(config.data.val))
    if split == "pool":
        spec.update(split="train", subset=(0, 0, 1_000_000))
    elif split == "test":
        spec.update(split="train", subset=(0, 1_000_000, None))
    elif split == "all":
        spec.update(split="all", subset=None, holdout_events_per_config=0)
    else:
        spec.update(split="val", subset=None)
    spec.update(loop=1, preload=False, index_workers=max(1, workers))
    if raw:
        spec["transform"] = []
    if data_root:
        spec["data_root"] = str(data_root)
        for source in spec.get("configs", []):
            if isinstance(source, dict):
                source.pop("data_root", None)
    if cache_dir:
        spec["index_cache_dir"] = str(cache_dir)
    dataset = build_dataset(spec)
    if max_events is not None:
        dataset.data_list = dataset.data_list[:max_events]
    return dataset


def _inference(args, features=False):
    import torch
    from pimm.datasets.utils import point_collate_fn
    from pimm.models import build_model
    from pimm.utils.checkpoints import checkpoint_model_state_dict
    from pimm.utils.config import Config

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = Config.fromfile(args.config)
    if features:
        config.data.val.update(max_per_source=-1, max_len=-1, max_events_per_config=-1)
        for source in config.data.val.configs:
            if isinstance(source, dict):
                source.pop("max_events", None)
        if "eval_transform" in config:
            config.data.val.transform = config.eval_transform
    dataset = make_dataset(config, args.split, args.data_root, args.cache_dir, args.max_events, args.workers)
    indices = np.array_split(np.arange(len(dataset)), args.shards)[args.shard]
    if not len(indices):
        raise ValueError("No events selected.")
    batch_size = args.batch_size or (16 if features else int(config.batch_size_val))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch_size,
                                         num_workers=args.workers, collate_fn=point_collate_fn)
    model = build_model(config.encoder if features else config.model).to(args.device).eval()
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_model_state_dict(raw)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    values, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if features:
                point = model({k: batch[k] for k in ("coord", "feat", "grid_coord", "offset")})
                bounds = torch.cat([point.offset.new_zeros(1), point.offset]).tolist()
                values.extend(point.feat[a:b].float().mean(0).cpu() for a, b in zip(bounds[:-1], bounds[1:]))
            else:
                logits = model(batch)["cls_logits"].float()
                values.extend((logits[:, 1] - logits[:, 0]).cpu())
            labels.append(batch["event_label"].view(-1).cpu())
    path = Path(args.out)
    if path.suffix != ".npz":
        path = Path(str(path) + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    value_key = "feat" if features else "score"
    np.savez(path, **{value_key: torch.stack(values).numpy()}, label=torch.cat(labels).numpy(),
             name=np.array([dataset.get_data_name(int(i)) for i in indices]), index=indices)
    if not features:
        _write_json(path.with_suffix(".json"), score_files([path], n=args.resamples, seed=args.seed))


def load_bank(paths):
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            rows.append({k: d[k] for k in d.files})
    keys = ("feat", "label", "name", "index")
    bank = {k: np.concatenate([d[k] for d in rows]) for k in keys if all(k in d for d in rows)}
    if "index" in bank:
        order = np.argsort(bank["index"])
        bank = {k: v[order] for k, v in bank.items()}
    if "name" not in bank:
        bank["name"] = np.arange(len(bank["label"])).astype(str)
    return bank


def rung(y, n, draw):
    if n == "all":
        return np.arange(len(y))
    n = int(n)
    if n == 100_000:
        return np.arange(n)
    k = n // 2
    return np.sort(np.concatenate([np.flatnonzero(y == c)[draw * k:(draw + 1) * k] for c in (0, 1)]))


def fit_logistic(pool, val, test=None, n="all", draw=0, threads=8):
    import torch
    from pimm.engines.hooks.eval.probes.events import logistic_cv

    torch.set_num_threads(min(threads, 16))
    idx = rung(pool["label"], n, draw)
    if n != "all" and len(idx) != int(n):
        raise ValueError("The requested label rung/draw is unavailable in the feature bank.")
    labels = pool["label"][idx]
    if min((labels == c).sum() for c in (0, 1)) < 5:
        raise ValueError("Five-fold logistic CV needs at least five fit events per class.")
    score_features = val["feat"] if test is None else np.concatenate([val["feat"], test["feat"]])
    xf = torch.from_numpy(pool["feat"][idx]).double()
    yf = torch.from_numpy(labels)
    xs = torch.from_numpy(score_features).double()
    scores, cs = zip(*(logistic_cv(xf, yf, xs, seed=s, folds=5) for s in range(3)))
    prob = np.mean(scores, axis=0)
    return prob[:len(val["label"])], None if test is None else prob[len(val["label"]):], list(cs), len(idx)


def _probe(args):
    pool, val = load_bank(args.pool), load_bank(args.val)
    test = load_bank(args.test) if args.test else None
    pv, pt, cs, count = fit_logistic(pool, val, test, args.labels, args.draw, args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "val.npz", score=pv, label=val["label"], name=val["name"])
    if test is not None:
        np.savez(out / "test.npz", score=pt, label=test["label"], name=test["name"])
    result = dict(n_labels=count, fold_seeds=[0, 1, 2], chosen_C=cs,
                  val=report([(pv, val["label"])], n=args.resamples, seed=args.seed))
    if test is not None:
        result["test"] = score_files([out / "test.npz"], [out / "val.npz"], n=args.resamples, seed=args.seed)
    _write_json(out / "metrics.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="Score stored logit margins or signal probabilities.")
    score.add_argument("--test", nargs="+", required=True)
    score.add_argument("--val", nargs="+")
    score.add_argument("--score-key", default="score")
    score.add_argument("--out", required=True)
    probe = sub.add_parser("probe", help="Fit the frozen mean-feature logistic probe.")
    probe.add_argument("--pool", nargs="+", required=True)
    probe.add_argument("--val", nargs="+", required=True)
    probe.add_argument("--test", nargs="+")
    probe.add_argument("--labels", default="all", help="Label count or all; low rungs are balanced.")
    probe.add_argument("--draw", type=int, default=0)
    probe.add_argument("--out", required=True)
    for command in (score, probe):
        command.add_argument("--resamples", type=int, default=1000)
        command.add_argument("--seed", type=int, default=0)
    probe.add_argument("--threads", type=int, choices=range(1, 17), default=8)
    for name in ("infer", "features"):
        command = sub.add_parser(name, help="Score a classifier." if name == "infer" else "Export mean-pooled frozen encoder features.")
        command.add_argument("--config", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--data-root")
        command.add_argument("--cache-dir")
        command.add_argument("--split", choices=("pool", "val", "test", "all"), default="test")
        command.add_argument("--batch-size", type=int, default=16 if name == "features" else None,
                             help="Defaults to 16 for features, config.batch_size_val for classifiers.")
        command.add_argument("--workers", type=int, choices=range(17), default=0)
        command.add_argument("--threads", type=int, choices=range(1, 17), default=8)
        command.add_argument("--max-events", type=int)
        command.add_argument("--shards", type=int, default=1)
        command.add_argument("--shard", type=int, default=0)
        command.add_argument("--device", default="cuda")
        command.add_argument("--out", required=True)
        command.add_argument("--resamples", type=int, default=1000)
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "score":
        result = score_files(args.test, args.val, args.resamples, args.seed, args.score_key)
        _write_json(args.out, result)
        print(json.dumps(result, indent=2))
    elif args.command == "probe":
        _probe(args)
    else:
        if args.shards < 1 or not 0 <= args.shard < args.shards:
            parser.error("--shard must be in [0, --shards).")
        _inference(args, features=args.command == "features")


if __name__ == "__main__":
    main()

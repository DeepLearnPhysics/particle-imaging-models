"""Two-blob baseline: python scripts/panda/eval/two_blob.py {extract,scan} --help."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

if __package__:
    from .next import at_threshold, make_dataset, report, val_threshold
else:
    from next import at_threshold, make_dataset, report, val_threshold

VOXELS = (5.0, 10.0)
RADII = (9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 36.0, 42.0, 48.0)
_DATASET = None
_GRID = None


def score_event(data, voxels=VOXELS, radii=RADII):
    from pimm.models.next.blobs import blob2_energy

    coord = data["coord"].astype(np.float64)
    energy = data["energy"][:, 0].astype(np.float64)
    return (int(data["event_label"][0]), data["name"], energy.sum(),
            np.stack([blob2_energy(coord, energy, radii, voxel) for voxel in voxels]))


def _score_index(index):
    return score_event(_DATASET.get_data(int(index)), *_GRID)


def scan(val, test=None, n=1000, seed=0):
    label, voxels, radii = val["label"], val["voxels"], val["radii"]
    result = {"energy only": report([(val["energy"], label)], n=n, seed=seed)}
    for i, voxel in enumerate(voxels):
        for j, radius in enumerate(radii):
            key = f"{voxel:g} mm voxels, R {radius:g} mm"
            result[key] = report([(val["e2"][:, i, j], label)], n=n, seed=seed)
    best = max((k for k in result if k != "energy only"), key=lambda k: result[k]["auroc"]["value"])
    i = list(voxels).index(float(best.split()[0]))
    j = list(radii).index(float(best.split()[-2]))
    out = dict(scan=result, best=best, voxel=float(voxels[i]), radius=float(radii[j]))
    if test is not None:
        if (label == 0).sum() < 10:
            raise ValueError("Validation FOM selection requires at least 10 background events.")
        ti = list(test["voxels"]).index(voxels[i])
        tj = list(test["radii"]).index(radii[j])
        scores = test["e2"][:, ti, tj].astype(np.float64)
        val_scores = val["e2"][:, i, j].astype(np.float64)
        threshold, at_val = val_threshold(val_scores, label)
        out["test"] = report([(scores, test["label"])], n=n, seed=seed)
        out["val_chosen_fom"] = dict(threshold=threshold, val=at_val,
                                     test=at_threshold(scores, test["label"], threshold, n=n, seed=seed))
    return out


def _extract(args):
    from pimm.utils.config import Config

    global _DATASET, _GRID
    config = Config.fromfile(args.config)
    dataset = make_dataset(config, args.split, args.data_root, args.cache_dir, args.max_events,
                           args.workers, raw=True)
    if not any(type(t).__name__ == "NEXTTopologyCut" for t in dataset.event_filter.transforms):
        raise ValueError("Two-blob extraction requires NEXTTopologyCut in the dataset event_filter config.")
    indices = np.array_split(np.arange(len(dataset)), args.shards)[args.shard]
    _DATASET, _GRID = dataset, (args.voxels, args.radii)
    if args.processes == 1:
        rows = [_score_index(i) for i in indices]
    else:
        with Pool(args.processes) as pool:
            rows = pool.map(_score_index, indices, chunksize=1)
    if not rows:
        raise ValueError("No events selected.")
    label, name, energy, e2 = map(np.array, zip(*rows))
    path = Path(args.out)
    if path.suffix != ".npz":
        path = Path(str(path) + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, label=label, name=name, energy=energy, e2=e2, index=indices,
             voxels=np.array(args.voxels), radii=np.array(args.radii))
    print(f"Scored {len(label)} events ({int((label == 1).sum())} signal).", flush=True)


def load_blobs(paths):
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            rows.append({k: d[k] for k in d.files})
    for d in rows[1:]:
        if not np.array_equal(d["voxels"], rows[0]["voxels"]) or not np.array_equal(d["radii"], rows[0]["radii"]):
            raise ValueError("Blob shards must have the same voxel/radius grid.")
    keys = ("label", "name", "energy", "e2", "index")
    merged = {k: np.concatenate([d[k] for d in rows]) for k in keys if all(k in d for d in rows)}
    if "index" in merged:
        order = np.argsort(merged["index"])
        merged = {k: v[order] for k, v in merged.items()}
    merged.update(voxels=rows[0]["voxels"], radii=rows[0]["radii"])
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Compute the weaker blob energy for each raw NEXT event.")
    extract.add_argument("--config", required=True)
    extract.add_argument("--data-root")
    extract.add_argument("--cache-dir")
    extract.add_argument("--split", choices=("pool", "val", "test", "all"), default="test")
    extract.add_argument("--workers", type=int, choices=range(17), default=0)
    extract.add_argument("--processes", type=int, choices=range(1, 17), default=1)
    extract.add_argument("--max-events", type=int)
    extract.add_argument("--voxels", type=float, nargs="+", default=VOXELS)
    extract.add_argument("--radii", type=float, nargs="+", default=RADII)
    extract.add_argument("--shards", type=int, default=1)
    extract.add_argument("--shard", type=int, default=0)
    extract.add_argument("--out", required=True)
    score = sub.add_parser("scan", help="Choose voxel size and blob radius on validation AUROC, then score test.")
    score.add_argument("--val", nargs="+", required=True)
    score.add_argument("--test", nargs="+")
    score.add_argument("--resamples", type=int, default=1000)
    score.add_argument("--seed", type=int, default=0)
    score.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "extract":
        if args.shards < 1 or not 0 <= args.shard < args.shards:
            parser.error("--shard must be in [0, --shards).")
        _extract(args)
    else:
        result = scan(load_blobs(args.val), load_blobs(args.test) if args.test else None,
                      args.resamples, args.seed)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

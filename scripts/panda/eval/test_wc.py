"""Score WC rings: python scripts/panda/eval/test_wc.py --config CONFIG --checkpoint WEIGHTS --out PREFIX."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pimm.datasets import build_dataset, point_collate_fn
from pimm.datasets.wand import PID_NAMES
from pimm.engines.hooks.eval.rings import ring_residuals

try:
    from ._common import load_config, load_model, save_json, seed, to_device
except ImportError:
    from _common import load_config, load_model, save_json, seed, to_device

SPECIES = {13: "mu-", -13: "mu+", 11: "e-", -11: "e+", 211: "pi+", -211: "pi-", 111: "pi0", 22: "gamma"}
IOU = 0.5
COORD_SCALE = 18.1
SIZE_BINS = (0, 300, 1000, 3000)
DECODING_RULE = ("max", 0.65, 0.5, 2, 50)
MICHEL_MAX_P = 60.0
MODEL_KEYS = ("coord", "grid_coord", "feat", "offset")


def decode_output(detector, output):
    # The paper scores stored float32 logits on CPU.
    output = {"outputs_by_label": {"particle": to_device(output["outputs_by_label"]["particle"], "cpu")}}
    rings = detector.postprocess(
        output,
        label="particle",
        conf_threshold=DECODING_RULE[1],
        mask_threshold=DECODING_RULE[2],
        min_points=DECODING_RULE[3],
        dedup_iou=1.0,
        decoding="wc",
    )[0]
    return sorted(rings, key=lambda r: (-r["score"], r["query"]))


def matches(pred, truth, weight=None):
    from scipy.optimize import linear_sum_assignment

    if not len(pred) or not len(truth):
        return np.zeros(0, int), np.zeros(0, int), np.zeros((len(pred), len(truth)))
    w = torch.ones(pred.shape[1]) if weight is None else weight.float().cpu()
    p, t = pred.float().cpu(), truth.float().cpu()
    inter = (p * w) @ t.T
    iou = (inter / ((p @ w)[:, None] + (t @ w)[None] - inter).clamp(min=1e-6)).numpy()
    k, m = iou.shape
    rank = (k - 1 - np.arange(k))[:, None] / k
    gain = (iou >= IOU) * (1 + (np.floor(iou * 1000) / 1000 + 1e-4 * rank) / (m + 1))
    rows, cols = linear_sum_assignment(-gain)
    keep = iou[rows, cols] >= IOU
    return rows[keep], cols[keep], iou


def valid_truth(truth):
    valid = truth["ring_valid"].bool()
    return {
        k: v[:, valid] if k == "inst_pe" else v[valid] if k.startswith("ring_") else v
        for k, v in truth.items()
    }


def score_event(event, rings, truth):
    truth = valid_truth(truth)
    pred = (
        torch.stack([r["mask"] for r in rings])
        if rings
        else torch.zeros((0, len(truth["inst_pe"])), dtype=torch.bool)
    )
    valid = truth["ring_valid"].bool()
    true = (truth["inst_pe"] > 0).T
    rows, cols, iou = matches(pred, true)
    q_rows, q_cols, q_iou = matches(pred, true, truth["inst_pe"].float().sum(1))
    true_pid = truth["ring_pid"].cpu()
    primaries = np.unique(truth["ring_anc_id"].cpu().numpy())
    species = SPECIES.get(int(truth["ring_anc_pdg"][0]), "other") if len(primaries) == 1 else None
    row = dict(
        event=event,
        points=int(true.shape[1]),
        det_tp=len(rows),
        det_pred=len(pred),
        det_truth=len(true),
        iou_sum=float(iou[rows, cols].sum()),
        pid_correct=int(sum(rings[r]["pid"] == int(true_pid[c]) for r, c in zip(rows, cols))),
        q_tp=len(q_rows),
        truth_by_size=np.bincount(
            np.digitize(true.sum(1).cpu().numpy(), SIZE_BINS) - 1, minlength=len(SIZE_BINS)
        ).tolist(),
        q_tp_by_size=(
            np.bincount(
                np.digitize(true.sum(1).cpu().numpy()[q_cols], SIZE_BINS) - 1, minlength=len(SIZE_BINS)
            ).tolist()
            if len(q_cols)
            else [0] * len(SIZE_BINS)
        ),
        q_pid_correct=int(sum(rings[r]["pid"] == int(true_pid[c]) for r, c in zip(q_rows, q_cols))),
        omega=float("nan"),
        primaries=len(primaries),
        species=species or ("none" if not len(primaries) else "multi"),
    )
    if rings and "direction" in rings[0] and "ring_direction" in truth:
        row.update(kinematics(rings, truth, valid, q_rows, q_cols))
    elif "ring_direction" in truth:
        row.update(kin_pid=[], kin_pred=[], dir_deg=[], mom_frac=[], vtx_m=[])
    return row


def event_records(event, rings, truth):
    truth = valid_truth(truth)
    row = score_event(event, rings, truth)
    true = (truth["inst_pe"] > 0).T
    match = {}
    if rings and len(true):
        rows, cols, iou = matches(torch.stack([r["mask"] for r in rings]), true, truth["inst_pe"].sum(1))
        match = {int(a): (int(b), float(iou[a, b])) for a, b in zip(rows, cols)}
    michel = (
        (truth["ring_pid"] == PID_NAMES.index("e"))
        & (truth["ring_momentum"] < np.log(MICHEL_MAX_P))
        & torch.isin(truth["ring_anc_pdg"].abs(), torch.tensor([13, 211]))
    ).tolist()
    unit = lambda v: (v / np.linalg.norm(v)).tolist()
    kept, truths = [], []
    for i, r in enumerate(rings):
        t, iou = match.get(i, (-1, float("nan")))
        kept.append(
            dict(
                event=int(event),
                pid=PID_NAMES[r["pid"]],
                **{f"p_{n}": float(x) for n, x in zip(PID_NAMES, r["cls_probs"])},
                score=r["score"],
                sites=int(r["mask"].sum()),
                p_MeV=float(np.exp(float(torch.as_tensor(r["momentum"]).reshape(-1)[0]))),
                **dict(
                    zip(
                        ("dir_x", "dir_y", "dir_z"),
                        unit(torch.as_tensor(r["direction"]).reshape(3).double().numpy()),
                    )
                ),
                **dict(
                    zip(
                        ("vtx_x_m", "vtx_y_m", "vtx_z_m"),
                        (torch.as_tensor(r["vertex"]).reshape(3).double().numpy() * COORD_SCALE).tolist(),
                    )
                ),
                truth_ring=t,
                iou=iou,
                ignored=bool(t >= 0 and michel[t]),
            )
        )
    for j in range(len(true)):
        truths.append(
            dict(
                event=int(event),
                ring=j,
                pid=PID_NAMES[int(truth["ring_pid"][j])],
                anc_id=int(truth["ring_anc_id"][j]),
                anc_pdg=int(truth["ring_anc_pdg"][j]),
                p_MeV=float(np.exp(float(truth["ring_momentum"][j]))),
                **dict(zip(("dir_x", "dir_y", "dir_z"), unit(truth["ring_direction"][j].double().numpy()))),
                **dict(
                    zip(
                        ("vtx_x_m", "vtx_y_m", "vtx_z_m"),
                        (truth["ring_vertex"][j].double().numpy() * COORD_SCALE).tolist(),
                    )
                ),
                vertex_valid=bool(truth["ring_vertex_valid"][j]),
                sites=int(true[j].sum()),
                pe=float(truth["inst_pe"][:, j].sum()),
                michel=bool(michel[j]),
            )
        )
    ignored = sum(1 for i in range(len(rings)) if i in match and michel[match[i][0]])
    row.update(
        n_pred_x=len(rings) - ignored,
        n_truth_x=len(true) - sum(michel),
        tp_x=sum(1 for i, (t, _) in match.items() if not michel[t]),
        pid_tp_x=sum(
            1 for i, (t, _) in match.items() if not michel[t] and rings[i]["pid"] == int(truth["ring_pid"][t])
        ),
    )
    return row, kept, truths


def kinematics(rings: list[dict], truth: dict, valid: torch.Tensor, rows, cols) -> dict:
    true = {
        key: truth[key][valid].cpu()
        for key in ("ring_pid", "ring_direction", "ring_momentum", "ring_vertex", "ring_vertex_valid")
    }
    out = dict(kin_pid=[], kin_pred=[], dir_deg=[], mom_frac=[], vtx_m=[])
    for r, c in zip(rows, cols):
        angle, frac, gap = ring_residuals(
            rings[r],
            true["ring_direction"][c],
            true["ring_momentum"][c],
            true["ring_vertex"][c],
            bool(true["ring_vertex_valid"][c]),
            COORD_SCALE,
        )
        out["dir_deg"].append(angle)
        out["mom_frac"].append(frac)
        out["vtx_m"].append(gap)
        out["kin_pid"].append(int(true["ring_pid"][c]))
        out["kin_pred"].append(int(rings[r]["pid"]))
    return out


def metrics(rows: list[dict]) -> dict:
    tp, pred, truth = (sum((r[k] for r in rows)) for k in ("det_tp", "det_pred", "det_truth"))
    return dict(
        events=len(rows),
        det_f1=2 * tp / max(pred + truth, 1),
        det_precision=tp / max(pred, 1),
        det_recall=tp / max(truth, 1),
        count_accuracy=float(np.mean([r["det_pred"] == r["det_truth"] for r in rows])),
        omega=float(np.mean([r["omega"] for r in rows])),
        mean_matched_iou=sum((r["iou_sum"] for r in rows)) / max(tp, 1),
        pid_acc=sum((r["pid_correct"] for r in rows)) / max(tp, 1),
        particle_f1=2 * sum((r["pid_correct"] for r in rows)) / max(pred + truth, 1),
        q_det_f1=2 * sum((r["q_tp"] for r in rows)) / max(pred + truth, 1),
        q_particle_f1=2 * sum((r["q_pid_correct"] for r in rows)) / max(pred + truth, 1),
        **{
            f"q_recall_sites_{lo}+" if hi is None else f"q_recall_sites_{lo}-{hi}": sum(
                (r["q_tp_by_size"][i] for r in rows)
            )
            / max(sum((r["truth_by_size"][i] for r in rows)), 1)
            for (i, (lo, hi)) in enumerate(zip(SIZE_BINS, SIZE_BINS[1:] + (None,)))
        },
        **kinematic_metrics(rows),
    )


def kinematic_metrics(rows: list[dict]) -> dict:
    if not rows or "dir_deg" not in rows[0]:
        return {}
    cat = lambda key: np.array(sum((r[key] for r in rows), []), np.float64)
    pid, direction, momentum, vertex = (cat("kin_pid"), cat("dir_deg"), cat("mom_frac"), cat("vtx_m"))
    q68 = lambda x: float(np.quantile(x[np.isfinite(x)], 0.68)) if np.isfinite(x).any() else float("nan")
    out = dict(
        direction_deg=float(np.nanmean(direction)) if len(direction) else float("nan"),
        direction_q68_deg=q68(direction),
        vertex_q68_m=q68(vertex),
        momentum_frac_q68=q68(momentum),
    )
    for name in ("e", "mu"):
        sel = pid == PID_NAMES.index(name)
        out.update(
            {
                f"vertex_{name}_q68_m": q68(vertex[sel]),
                f"momentum_{name}_frac_q68": q68(momentum[sel]),
                f"direction_{name}_q68_deg": q68(direction[sel]),
            }
        )
    return out


def splits(rows: list[dict]) -> dict[str, list[int]]:
    primaries = lambda r: "1" if r["primaries"] <= 1 else "2+"
    rings = lambda r: str(r["det_truth"]) if r["det_truth"] < 3 else "3+"
    keep = lambda test: [i for (i, r) in enumerate(rows) if test(r)]
    out = {"all": list(range(len(rows)))}
    for p in ("1", "2+"):
        out[f"primaries={p}"] = keep(lambda r: primaries(r) == p)
        for n in ("0", "1", "2", "3+"):
            out[f"primaries={p},rings={n}"] = keep(lambda r: primaries(r) == p and rings(r) == n)
    for n in ("0", "1", "2", "3+"):
        out[f"rings={n}"] = keep(lambda r: rings(r) == n)
    for s in sorted({r["species"] for r in rows}):
        out[f"species={s}"] = keep(lambda r: r["species"] == s)
    return {name: index for (name, index) in out.items() if index}


def summarize(rows):
    out = {name: metrics([rows[i] for i in index]) for name, index in splits(rows).items()}
    for name, lo, hi in (("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, 10**9)):
        sel = [r for r in rows if lo <= r["det_truth"] <= hi]
        if sel:
            out[f"rings={name}"] = metrics(sel)
    groups = dict(
        splits(rows),
        **{
            k: [i for (i, r) in enumerate(rows) if lo <= r["det_truth"] <= hi]
            for (k, lo, hi) in (("rings=3-4", 3, 4), ("rings=5-9", 5, 9), ("rings=10+", 10, 10**9))
        },
    )
    for name, m in out.items():
        sel = [rows[i] for i in groups[name]]
        tp = sum((r["q_tp"] for r in sel))
        m.update(
            q_det_precision=tp / max(sum((r["det_pred"] for r in sel)), 1),
            q_det_recall=tp / max(sum((r["det_truth"] for r in sel)), 1),
        )
    return out


def summarize_x(rows, rings, truths):
    import pandas as pd

    r = pd.DataFrame(rows)
    m = rings[(rings.truth_ring >= 0) & ~rings.ignored].merge(
        truths, left_on=["event", "truth_ring"], right_on=["event", "ring"], suffixes=("", "_t")
    )
    ang = np.degrees(
        np.arccos(
            np.clip(
                (m[["dir_x", "dir_y", "dir_z"]].values * m[["dir_x_t", "dir_y_t", "dir_z_t"]].values).sum(1),
                -1,
                1,
            )
        )
    )
    gap = np.linalg.norm(
        m[["vtx_x_m", "vtx_y_m", "vtx_z_m"]].values - m[["vtx_x_m_t", "vtx_y_m_t", "vtx_z_m_t"]].values,
        axis=1,
    )
    m = m.assign(
        ang=ang, gap=np.where(m.vertex_valid, gap, np.nan), dp=np.abs(m.p_MeV - m.p_MeV_t) / m.p_MeV_t
    )

    def q68(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanquantile(x, 0.68)) if np.isfinite(x).any() else float("nan")

    bins = {"1": (1, 1), "2": (2, 2), "3-4": (3, 4), "5-9": (5, 9), "10+": (10, 10**9)}
    splits = {
        "all": r.index,
        **{f"species={s}": r.index[r.species == s] for s in sorted(r.species.unique())},
        **{
            f"rings={b}": r.index[(r.n_truth_x >= lo) & (r.n_truth_x <= hi)] for (b, (lo, hi)) in bins.items()
        },
        "rings=0": r.index[r.n_truth_x == 0],
    }
    out = {}
    for name, idx in splits.items():
        if not len(idx):
            continue
        g = r.loc[idx]
        tp, pred, true, pid = (g.tp_x.sum(), g.n_pred_x.sum(), g.n_truth_x.sum(), g.pid_tp_x.sum())
        mm = m[m.event.isin(g.event)]
        k = {}
        for pid_name in ("e", "mu", "gamma", "pi", "all"):
            x = mm if pid_name == "all" else mm[mm.pid_t == pid_name]
            k.update(
                {
                    f"direction_{pid_name}_q68_deg": q68(x.ang.values),
                    f"vertex_{pid_name}_q68_m": q68(x.gap.values),
                    f"momentum_{pid_name}_frac_q68": q68(x.dp.values),
                    f"matched_{pid_name}": int(len(x)),
                }
            )
        out[name] = dict(
            events=int(len(g)),
            over=float((g.n_pred_x > g.n_truth_x).mean()),
            under=float((g.n_pred_x < g.n_truth_x).mean()),
            exact=float((g.n_pred_x == g.n_truth_x).mean()),
            ring_precision=tp / max(pred, 1),
            ring_recall=tp / max(true, 1),
            ring_f1=2 * tp / max(pred + true, 1),
            particle_f1=2 * pid / max(pred + true, 1),
            kept_rings=int(pred),
            truth_rings=int(true),
            **k,
        )
    return out


def count_confusion(rows, top=10, keys=("det_truth", "det_pred")):
    m = np.zeros((top + 1, top + 1), np.int64)
    for r in rows:
        m[min(r[keys[0]], top), min(r[keys[1]], top)] += 1
    return dict(labels=[str(i) for i in range(top)] + [f"{top}+"], rows_true_cols_kept=m.tolist())


RING_COLUMNS = [
    "event",
    "pid",
    "score",
    "sites",
    "p_MeV",
    "dir_x",
    "dir_y",
    "dir_z",
    "vtx_x_m",
    "vtx_y_m",
    "vtx_z_m",
    "truth_ring",
    "iou",
    "ignored",
]
TRUTH_COLUMNS = [
    "event",
    "ring",
    "pid",
    "anc_id",
    "anc_pdg",
    "p_MeV",
    "dir_x",
    "dir_y",
    "dir_z",
    "vtx_x_m",
    "vtx_y_m",
    "vtx_z_m",
    "vertex_valid",
    "sites",
    "pe",
    "michel",
]


def summarize_records(rows, rings, truths):
    ring_table = pd.DataFrame(rings) if rings else pd.DataFrame(columns=RING_COLUMNS)
    truth_table = pd.DataFrame(truths) if truths else pd.DataFrame(columns=TRUTH_COLUMNS)
    summary_x = summarize_x(rows, ring_table, truth_table)
    species = sorted({r["species"] for r in rows})
    confusion = {
        "all": count_confusion(rows),
        **{f"species={s}": count_confusion([r for r in rows if r["species"] == s]) for s in species},
    }
    keys = ("n_truth_x", "n_pred_x")
    confusion_x = {
        "all": count_confusion(rows, keys=keys),
        **{
            f"species={s}": count_confusion([r for r in rows if r["species"] == s], keys=keys)
            for s in species
        },
    }
    return dict(
        rule=DECODING_RULE,
        events=len(rows),
        michel_excluded=dict(
            definition=f"e ring, p < {MICHEL_MAX_P} MeV/c, mu / pi primary ancestor; matched kept rings ignored",
            summary=summary_x,
            count_confusion=confusion_x,
        ),
        all_rings=dict(summary=summarize(rows), count_confusion=confusion),
    )


@torch.no_grad()
def score_run(args):
    seed(args.seed)
    cfg = load_config(args.config)
    model = load_model(cfg, args.checkpoint, device=args.device, disable_flash=True)
    core = model.model if getattr(model, "model", None) is not None else model
    if getattr(core, "average_placements", False):
        raise ValueError("WC evaluation requires a single forward placement")
    spec = copy.deepcopy(dict(cfg.data.get("val", cfg.data.get("test"))))
    spec.update(
        split=args.split,
        config=args.wc_config,
        first_event=args.first_event,
        loop=1,
        test_mode=False,
        max_len=-1 if args.max_events is None else args.max_events,
    )
    if args.data_root is not None:
        spec["data_root"] = str(args.data_root)
    dataset = build_dataset(spec)
    batches = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=point_collate_fn)
    torch.manual_seed(args.seed)
    rows, rings, truths = [], [], []
    for event, batch in enumerate(batches):
        output = model(to_device({k: batch[k] for k in MODEL_KEYS}, args.device))
        decoded = decode_output(model, output)
        row, kept, truth = event_records(event, decoded, batch)
        name = batch["name"][0]
        row["name"] = name
        for record in kept + truth:
            record["name"] = name
        rows.append(row)
        rings.extend(kept)
        truths.extend(truth)
    if not rows:
        raise ValueError("The selected WC split contains no events")
    result = summarize_records(rows, rings, truths)
    result.update(split=args.split, config=args.wc_config, checkpoint=str(args.checkpoint))
    save_json(str(args.out) + ".json", result)
    save_json(str(args.out) + ".rows.json", rows)
    pd.DataFrame(
        rings, columns=list(dict.fromkeys(RING_COLUMNS + [f"p_{n}" for n in PID_NAMES] + ["name"]))
    ).to_parquet(str(args.out) + ".rings.parquet", index=False)
    pd.DataFrame(truths, columns=TRUTH_COLUMNS + ["name"]).to_parquet(
        str(args.out) + ".truth.parquet", index=False
    )
    print(json.dumps(result["michel_excluded"]["summary"]["all"], indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--wc-config", choices=("config_01", "multi_primary"), default="config_01")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--first-event", type=int, default=0)
    parser.add_argument("--max-events", type=int, help="Number of events to score")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    score_run(args)


if __name__ == "__main__":
    main()

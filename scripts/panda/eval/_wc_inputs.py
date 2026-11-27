"""WC probe preparation, invoked by python scripts/panda/eval/wc_probes.py prepare --help."""

import glob
import io
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RADIUS_M, HALF_Z_M, WALL_M = 16.9, 18.1, 2.0
SPLITS = {"train": 0, "val": 1}
SPLIT_CODE = {"val": 1, "test": 2}
PREP_SPECIES = {"e": 11, "mu": 13, "pi": 211, "pi0": 111, "gamma": 22}
PDG_SPECIES = {code: name for name, code in PREP_SPECIES.items()}
PDG_MASS_MEV = {11: 0.51099895, 13: 105.6583755, 211: 139.57039, 111: 134.9768, 22: 0.0}
INPUT_KEYS = (("sensor_idx", np.int32), ("PE", np.float32), ("T", np.float32))


def map_tasks(function, tasks, workers, chunksize=1):
    if workers < 2:
        yield from map(function, tasks)
    else:
        with ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            yield from pool.map(function, tasks, chunksize=chunksize)


def dataset_events(data_root, split):
    from pimm.datasets.wand import WANDRingSitesDataset

    events = WANDRingSitesDataset(
        data_root, split=split, all_train_events=split == "train"
    ).reader.events
    index = np.load(os.path.join(data_root, "metadata", "event_index.npz"))
    rows = np.flatnonzero(index["split_code"] == SPLITS[split])
    if split == "train":
        rows = rows[np.argsort(index["file_index"][rows], kind="stable")]
        file = index["file_index"][rows]
        number = np.arange(len(rows)) - np.searchsorted(file, file)
        lookup = {
            (f, n): r
            for (f, n, r) in zip(file.tolist(), number.tolist(), rows.tolist())
        }
        rows = np.array([lookup[f, n] for (f, n) in events.tolist()])
    assert len(rows) == len(events), (split, len(rows), len(events))
    return (events, rows)


def check_file(args):
    path, numbers, expect = args
    out = []
    with open(path, "rb") as raw:
        data = io.BytesIO(raw.read())
    with h5py.File(data, "r") as f:
        for number, (source, file, event, pdg, ke, direction, vertex) in zip(
            numbers, expect
        ):
            g = f[f"event_{number:03d}"]
            a = g.attrs
            if (
                a["source_event_idx"],
                a["original_file_index"],
                a["original_event_num"],
            ) != (source, file, event):
                out.append(None)
                continue
            track, inter = (g["per_track"], g["per_interaction"])
            primary = np.flatnonzero(track["parent_id"][:] == 0)
            p = primary[0]
            d = np.array([track[f"dir_{x}"][p] for x in "xyz"], np.float64)
            v = np.array([inter[f"vertex_{x}"][0] for x in "xyz"], np.float64)
            out.append(
                dict(
                    contained=bool(g["per_event/contained"][()]),
                    primaries=len(primary),
                    pdg_ok=int(track["pdg"][p]) == pdg,
                    dke=abs(float(track["initial_energy"][p]) - ke),
                    dangle=float(
                        np.degrees(
                            np.arccos(np.clip(d @ direction / np.linalg.norm(d), -1, 1))
                        )
                    ),
                    dvertex=float(np.abs(v - vertex).max()),
                )
            )
    return out


def prepare_pools(data_root, workers=0, splits=None, max_events=None):
    from pimm.engines.hooks.eval.probes.wc import fiducial_mask

    index = dict(np.load(os.path.join(data_root, "metadata", "event_index.npz")))
    first = index["primary_offsets"][:-1].clip(max=len(index["primary_pdg"]) - 1)
    pdg, ke, direction = (
        index["primary_pdg"][first],
        index["primary_ke_MeV"][first],
        index["primary_dir"][first],
    )
    vertex = index["vertex"].astype(np.float64)
    wall = fiducial_mask(
        vertex, margin_m=WALL_M, radius_m=RADIUS_M, half_z_m=HALF_Z_M
    )
    pools, counts, worst = ({}, {}, {})
    for split in (SPLITS if splits is None else splits):
        events, rows = dataset_events(data_root, split)
        if max_events is not None:
            events, rows = events[:max_events], rows[:max_events]
        single = index["n_primaries"][rows] == 1
        species = np.abs(pdg[rows])
        candidate = np.flatnonzero(
            single & wall[rows] & np.isin(species, list(PREP_SPECIES.values()))
        )
        jobs = {}
        for position in candidate:
            jobs.setdefault(int(events[position, 0]), []).append(position)
        tasks = []
        for file, positions in sorted(jobs.items()):
            r = rows[positions]
            expect = list(
                zip(
                    index["source_event_idx"][r].tolist(),
                    index["file_index"][r].tolist(),
                    index["event_num"][r].tolist(),
                    pdg[r].tolist(),
                    ke[r].astype(float).tolist(),
                    direction[r].astype(float),
                    vertex[r],
                )
            )
            tasks.append(
                (
                    os.path.join(
                        data_root, split, "config_01", "labl", f"wc_labl_{file:04d}.h5"
                    ),
                    events[positions, 1].tolist(),
                    expect,
                )
            )
        results = [
            x for chunk in map_tasks(check_file, tasks, workers, 8) for x in chunk
        ]
        order = (
            np.concatenate([positions for (_, positions) in sorted(jobs.items())])
            if jobs
            else np.empty(0, np.int64)
        )
        assert all(
            (r is not None for r in results)
        ), f"{split}: a labl group does not match its index row"
        assert all(
            (r["primaries"] == 1 and r["pdg_ok"] for r in results)
        ), f"{split}: primary track mismatch"
        contained = dict(zip(order.tolist(), [r["contained"] for r in results]))
        worst[split] = {
            k: max((r[k] for r in results), default=0.0)
            for k in ("dke", "dangle", "dvertex")
        }
        counts[split] = dict(events=len(rows), single=int(single.sum()))
        for name, code in PREP_SPECIES.items():
            keep = [p for p in candidate if species[p] == code and contained[int(p)]]
            keep = np.array(keep, np.int64)
            assert (
                index["hit_pmts"][rows[keep]] > 0
            ).all(), f"{split} {name}: an event has no hit PMT"
            pools[f"{split}_{name}_file"], pools[f"{split}_{name}_event"] = (
                events[keep, 0],
                events[keep, 1],
            )
            pools[f"{split}_{name}_row"] = rows[keep]
            counts[split][name] = dict(
                single=int((single & (species == code)).sum()),
                wall=int((single & wall[rows] & (species == code)).sum()),
                contained=len(keep),
            )
            print(split, name, counts[split][name], flush=True)
        print(
            split,
            "truth check, worst |dKE| MeV, angle deg, |dvertex| m:",
            worst[split],
            flush=True,
        )
    return (
        pools,
        dict(
            data_root=data_root,
            wall_m=WALL_M,
            radius_m=RADIUS_M,
            half_z_m=HALF_Z_M,
            counts=counts,
            truth_check_worst=worst,
        ),
    )


def dwall(v):
    return np.minimum(RADIUS_M - np.hypot(v[:, 0], v[:, 1]), HALF_Z_M - np.abs(v[:, 2]))


def towall(v, d):
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (v[:, 0] * d[:, 0] + v[:, 1] * d[:, 1])
    c = v[:, 0] ** 2 + v[:, 1] ** 2 - RADIUS_M**2
    disc = np.sqrt(np.maximum(b**2 - 4 * a * c, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        side = np.where(a > 0, (-b + disc) / (2 * a), np.inf)
        cap = np.where(
            d[:, 2] != 0, (np.sign(d[:, 2]) * HALF_Z_M - v[:, 2]) / d[:, 2], np.inf
        )
    return np.minimum(np.where(side > 0, side, np.inf), np.where(cap > 0, cap, np.inf))


def read_truth_file(data_root, split, file, numbers):
    out = []
    groups = {}
    for kind in ("labl", "sensor"):
        with open(
            f"{data_root}/{split}/config_01/{kind}/wc_{kind}_{file:04d}.h5", "rb"
        ) as raw:
            groups[kind] = h5py.File(io.BytesIO(raw.read()), "r")
    for n in numbers:
        g, s = (groups["labl"][f"event_{n:03d}"], groups["sensor"][f"event_{n:03d}"])
        track = g["per_track"]
        primary = np.flatnonzero(track["parent_id"][:] == 0)
        assert len(primary) == 1, (file, n, "primaries")
        p = primary[0]
        part = int(track["particle_idx"][p])
        pe, sensor = (s["PE"][:], s["sensor_idx"][:])
        out.append(
            dict(
                source_event_idx=int(g.attrs["source_event_idx"]),
                original_file=int(g.attrs["original_file_index"]),
                original_event=int(g.attrs["original_event_num"]),
                labl_pdg=int(track["pdg"][p]),
                labl_ke_MeV=float(track["initial_energy"][p]),
                labl_dir=np.array([track[f"dir_{x}"][p] for x in "xyz"], np.float64),
                labl_vertex=np.array(
                    [g["per_interaction"][f"vertex_{x}"][0] for x in "xyz"], np.float64
                ),
                contained=bool(g["per_event/contained"][()]),
                primary_contained=bool(g["per_particle/contained"][part]),
                primary_n_cherenkov=int(track["n_cherenkov"][p]),
                total_n_cherenkov=int(track["n_cherenkov"][:].sum()),
                n_tracks=len(track["pdg"]),
                t0_ns=float(g["per_event/t0"][()]),
                n_digits=len(pe),
                n_hit_sensors=len(np.unique(sensor)),
                total_pe=float(pe.sum()),
            )
        )
    for h in groups.values():
        h.close()
    return out


def prepare_truth(data_root, split, max_events=None):
    index = np.load(f"{data_root}/metadata/event_index.npz")
    rows = np.flatnonzero(index["split_code"] == SPLIT_CODE[split])
    files = sorted(
        (
            int(os.path.basename(f)[8:12])
            for f in glob.glob(f"{data_root}/{split}/config_01/labl/*.h5")
        )
    )
    counts = []
    for file in files:
        with h5py.File(
            f"{data_root}/{split}/config_01/labl/wc_labl_{file:04d}.h5", "r"
        ) as f:
            counts.append(sum((k.startswith("event_") for k in f)))
    assert sum(counts) == len(rows), (sum(counts), len(rows))
    file_of = np.repeat(files, counts)
    number = np.concatenate([np.arange(c) for c in counts])
    if max_events is not None:
        rows, file_of, number = (
            rows[:max_events],
            file_of[:max_events],
            number[:max_events],
        )
    labl = [
        x
        for file in dict.fromkeys(file_of.tolist())
        for x in read_truth_file(data_root, split, file, number[file_of == file])
    ]
    first = index["primary_offsets"][rows]
    pdg = index["primary_pdg"][first].astype(np.int64)
    ke = index["primary_ke_MeV"][first].astype(np.float64)
    direction = index["primary_dir"][first].astype(np.float64)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    vertex = index["vertex"][rows].astype(np.float64)
    for k, (x, r) in enumerate(zip(labl, rows)):
        assert (x["source_event_idx"], x["original_file"], x["original_event"]) == (
            index["source_event_idx"][r],
            index["file_index"][r],
            index["event_num"][r],
        ), (split, k, "identity")
        assert x["labl_pdg"] == pdg[k] and abs(x["labl_ke_MeV"] - ke[k]) < 0.01 * max(
            1.0, ke[k]
        ), (split, k, "primary")
        assert (
            np.degrees(
                np.arccos(
                    np.clip(
                        x["labl_dir"] @ direction[k] / np.linalg.norm(x["labl_dir"]),
                        -1,
                        1,
                    )
                )
            )
            < 0.1
        )
        assert np.abs(x["labl_vertex"] - vertex[k]).max() < 0.001, (split, k, "vertex")
    assert (index["n_primaries"][rows] == 1).all()
    mass = np.array([PDG_MASS_MEV[abs(p)] for p in pdg])
    df = pd.DataFrame(
        dict(
            split=split,
            file=file_of,
            event=number,
            index_row=rows,
            source_event_idx=index["source_event_idx"][rows],
            pdg=pdg,
            species=[PDG_SPECIES[abs(p)] for p in pdg],
            charge=np.sign(pdg) * np.where(np.isin(np.abs(pdg), (111, 22)), 0, 1),
            ke_MeV=ke,
            mass_MeV=mass,
            p_MeV=np.sqrt(ke * (ke + 2 * mass)),
            dir_x=direction[:, 0],
            dir_y=direction[:, 1],
            dir_z=direction[:, 2],
            vtx_x_m=vertex[:, 0],
            vtx_y_m=vertex[:, 1],
            vtx_z_m=vertex[:, 2],
            vtx_r_m=np.hypot(vertex[:, 0], vertex[:, 1]),
            dwall_m=dwall(vertex),
            towall_m=towall(vertex, direction),
            hit_pmts=index["hit_pmts"][rows],
            **{
                k: [x[k] for x in labl]
                for k in (
                    "contained",
                    "primary_contained",
                    "primary_n_cherenkov",
                    "total_n_cherenkov",
                    "n_tracks",
                    "t0_ns",
                    "n_digits",
                    "n_hit_sensors",
                    "total_pe",
                )
            },
        )
    )
    df["charge"] = np.select(
        [np.isin(df.pdg, (11, 13, -211)), np.isin(df.pdg, (-11, -13, 211))], [-1, 1], 0
    )
    return df


def read_input_file(args):
    path, numbers = args
    with open(path, "rb") as raw:
        data = io.BytesIO(raw.read())
    with h5py.File(data, "r") as f:
        return [
            tuple(
                (
                    f[f"event_{n:03d}"][key][:].astype(dtype)
                    for (key, dtype) in INPUT_KEYS
                )
            )
            for n in numbers
        ]


def prepare_inputs(data_root, split, events, out, workers):
    by_file = {}
    for i, (file, number) in enumerate(events.tolist()):
        by_file.setdefault(file, []).append((i, number))
    tasks = [
        (
            str(
                Path(data_root)
                / split
                / "config_01/sensor"
                / f"wc_sensor_{file:04d}.h5"
            ),
            [number for _, number in rows],
        )
        for file, rows in sorted(by_file.items())
    ]
    digits = [None] * len(events)
    if workers == 0:
        pieces = map(read_input_file, tasks)
        for (_, rows), piece in zip(sorted(by_file.items()), pieces):
            for (i, _), values in zip(rows, piece):
                digits[i] = values
    else:
        with ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for (_, rows), piece in zip(
                sorted(by_file.items()), pool.map(read_input_file, tasks, chunksize=4)
            ):
                for (i, _), values in zip(rows, piece):
                    digits[i] = values
    lengths = np.array([len(values[0]) for values in digits])
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path) + ".tmp", "w") as handle:
        handle.attrs["data_root"], handle.attrs["split"] = str(data_root), split
        handle["file"], handle["event"] = events[:, 0], events[:, 1]
        handle["offsets"] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        for k, (key, _) in enumerate(INPUT_KEYS):
            handle[key] = np.concatenate([values[k] for values in digits])
    os.replace(str(path) + ".tmp", path)
    print(
        json.dumps(
            dict(
                split=split,
                events=len(events),
                digits=int(lengths.sum()),
                out=str(path),
            )
        ),
        flush=True,
    )


def prepare(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.max_events is not None and args.max_events < 1:
        raise ValueError("--max-events must be positive")
    if args.stage == "pools":
        if args.split == "test":
            raise ValueError("Probe training pools use train and val only")
        pools, _ = prepare_pools(
            args.data_root,
            args.workers,
            None if args.split is None else (args.split,),
            args.max_events,
        )
        with open(str(out) + ".tmp", "wb") as handle:
            np.savez(handle, **pools)
        os.replace(str(out) + ".tmp", out)
    elif args.stage == "truth":
        if args.split not in ("val", "test"):
            raise ValueError("Truth preparation requires --split val or test")
        table = prepare_truth(args.data_root, args.split, args.max_events)
        table.to_parquet(str(out) + ".tmp", index=False)
        os.replace(str(out) + ".tmp", out)
        print(f"{len(table)} images -> {out}", flush=True)
    else:
        if args.split is None:
            raise ValueError("Input preparation requires --split")
        if bool(args.truth) == bool(args.pools):
            raise ValueError("Provide exactly one of --truth or --pools")
        if args.truth:
            table = pd.read_parquet(args.truth)
            events = np.stack([table.file.values, table.event.values], 1)
        else:
            if args.split == "test":
                raise ValueError("Test input preparation uses --truth")
            with np.load(args.pools) as pools:
                count = args.per_species if args.split == "train" else None
                events = np.concatenate(
                    [
                        np.stack(
                            [
                                pools[f"{args.split}_{s}_file"][:count],
                                pools[f"{args.split}_{s}_event"][:count],
                            ],
                            1,
                        )
                        for s in PREP_SPECIES
                    ]
                )
        if args.max_events is not None:
            events = events[: args.max_events]
        if not len(events):
            raise ValueError("No images selected")
        prepare_inputs(args.data_root, args.split, events, out, args.workers)


def add_parser(subcommands):
    parser = subcommands.add_parser(
        "prepare", help="Regenerate probe pools, truth tables or compact digit inputs"
    )
    parser.add_argument("--stage", choices=("pools", "truth", "inputs"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        help="Pools default to both train and val",
    )
    parser.add_argument("--pools", help="Pool NPZ for train/val compact inputs")
    parser.add_argument("--truth", help="Truth parquet for val/test compact inputs")
    parser.add_argument(
        "--per-species",
        type=int,
        default=10000,
        help="Training prefix copied per species",
    )
    parser.add_argument("--workers", type=int, choices=range(17), default=4)
    parser.add_argument(
        "--max-events", type=int, help="Bound the number of images for a check"
    )
    parser.add_argument("--out", required=True, help="Output NPZ, parquet or HDF5 file")
    parser.set_defaults(prepare_handler=prepare)
    return parser

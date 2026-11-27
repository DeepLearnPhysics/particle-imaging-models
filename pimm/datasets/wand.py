"""Water-Cherenkov rings on hit sites of the PMT array.

A PMT's charge is summed and its time is the earliest over its digits. A site is a voxel of the PMT table's
lattice, with ``grid_size`` in units of ``coord / coord_scale``. Hit PMTs in one voxel contribute summed charge
and the earliest time. Its position is a hit PMT drawn with ``np.random.default_rng(seed)`` in site-key order.
Times are measured relative to first light. Transforms construct log-charge and Fourier-time features;
``hit`` (1) and ``hidden`` (0) are constant channels.

Digit-to-particle truth is summed onto sites. A ring lights at least
``min_ring_hits`` PMTs. A site belongs to every ring contributing charge, and unlit sites are unowned.
``ring_anc_id`` and ``ring_anc_pdg`` identify each ring's primary ancestor. Ring kinematics contain log momentum,
initial direction, and normalized vertex, valid for primaries and photons from a primary neutral pion.

Files follow ``<split>/<config>/{sensor,hits,labl}/wc_<modality>_NNNN.h5``. Training reads
``metadata/subset_train_over1024.npy`` (file, event) rows in seed-0 order, with the row index as the site seed.
``all_train_events`` reads all training events from ``metadata/event_index.npz`` in seed-0 shuffled order.
Validation and test events follow file order.

``max_len`` keeps the first events. Overlay draws a donor distinct from the host until the rings fit
``max_rings``. Each event is shifted to its own first light before hit PMTs are merged and sites are built.
"""

import os
from collections import OrderedDict

import h5py
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pilarnet.overlay import PILArNetOverlayMixin
from .transform import Compose
from .transform.spatial import GridSample

_PDG_MASS_MEV = {
    11: 0.51099895,
    -11: 0.51099895,
    13: 105.6583755,
    -13: 105.6583755,
    22: 0.0,
    111: 134.9768,
    211: 139.57039,
    -211: 139.57039,
    2212: 938.27208816,
    2112: 939.5654205,
    321: 493.677,
    -321: 493.677,
}

PID_NAMES = ("e", "gamma", "mu", "pi", "p", "n", "K", "other")


def pdg_to_pid(pdg: int) -> int:
    a = abs(int(pdg))
    if a == 11:
        return 0  # e+/e-
    if pdg == 22:
        return 1  # gamma
    if a == 13:
        return 2  # mu+/mu-
    if a == 211:
        return 3  # pi+/pi-
    if pdg == 2212:
        return 4  # proton
    if pdg == 2112:
        return 5  # neutron
    if a == 321:
        return 6  # K+/K-
    return 7  # other


class _RingTruth(Dataset):
    def _particle_truth(self, labl, event_idx):
        """Resolve each particle's visible radiator and initial kinematics."""
        L = labl.read_event(int(event_idx))
        cats = L.get("labl_particle_category")
        gdata = L.get("labl_particle_genealogy_data")
        goff = L.get("labl_particle_genealogy_offsets")
        track_id = L.get("labl_track_track_id")
        pdg = L.get("labl_track_pdg")
        e0 = L.get("labl_track_initial_energy")
        tpi = L.get("labl_track_particle_idx")  # track -> per_particle FK
        ncher = L.get("labl_track_n_cherenkov")  # Cherenkov photons per track
        parent_id = L.get("labl_track_parent_id")  # track -> parent track_id
        if cats is None or gdata is None or goff is None or track_id is None:
            return {}
        id2row = {int(t): i for i, t in enumerate(track_id)}
        P = len(cats)

        t_dir = [L.get(f"labl_track_dir_{a}") for a in "xyz"]
        t_int = L.get("labl_track_interaction")
        i_vtx = [L.get(f"labl_interaction_vertex_{a}") for a in "xyz"]

        def _starts_at_vertex(row):
            if parent_id is None or int(parent_id[row]) == 0:
                return parent_id is not None
            prow = id2row.get(int(parent_id[row]), -1)
            return prow >= 0 and int(pdg[prow]) == 111 and int(parent_id[prow]) == 0

        truth = {}
        for k in range(P):
            root_tid = int(gdata[int(goff[k + 1]) - 1])
            row = id2row.get(root_tid, -1)
            if row < 0:
                continue
            anc_row = id2row.get(int(gdata[int(goff[k])]), -1)
            anc_pdg = int(pdg[anc_row]) if anc_row >= 0 else int(pdg[row])
            own_ncher = int(ncher[row]) if ncher is not None else 1
            # Non-radiating hadrons inherit their dominant visible descendant.
            if (
                own_ncher <= 0
                and int(pdg[row]) != 22
                and tpi is not None
                and ncher is not None
            ):
                trk = np.where(tpi == k)[0]
                if trk.size and int(ncher[trk].max()) > 0:
                    row = int(trk[int(np.argmax(ncher[trk]))])
                else:
                    continue  # no Cherenkov-producing track -> invisible
            pdg_k = int(pdg[row])
            ke_k = float(e0[row]) if e0 is not None else float("nan")
            vtx = None
            dvec = None
            if t_dir[0] is not None:
                d = np.array([t_dir[a][row] for a in range(3)], dtype=np.float32)
                n = float(np.linalg.norm(d))
                dvec = (
                    (d / n).astype(np.float32) if np.isfinite(n) and n > 1e-6 else None
                )
            if i_vtx[0] is not None and t_int is not None and _starts_at_vertex(row):
                vtx = np.array(
                    [i_vtx[a][int(t_int[row])] for a in range(3)], dtype=np.float32
                )
            truth[k] = dict(
                pdg=pdg_k,
                ke=ke_k,
                vertex=vtx,
                direction=dvec,
                anc_pdg=anc_pdg,
                anc_id=int(gdata[int(goff[k])]),
            )
        return truth

    def get_data(self, idx):
        source_idx, event_idx = self.data_list[idx % len(self.data_list)]
        src = self.sources[source_idx]
        sample = src["dataset"][event_idx]
        sensor = sample["sensor"]
        hits = sample["hits"]

        coord = np.asarray(sensor["coord"], dtype=np.float32)  # (N, 3)
        energy = np.asarray(sensor["energy"], dtype=np.float32).reshape(-1, 1)
        time = np.asarray(sensor["time"], dtype=np.float32).reshape(-1, 1)
        sensor_idx = np.asarray(sensor["sensor_idx"], dtype=np.int64)  # (N,)
        N = coord.shape[0]

        if N < self.min_points:
            raise ValueError(f"WAND event {idx} has no sensor digits")

        h_sensor = np.asarray(hits["sensor_idx"], dtype=np.int64)
        h_part = np.asarray(hits["particle_idx"], dtype=np.int64)
        h_pe = np.asarray(hits["energy"], dtype=np.float32).reshape(-1)

        # Digit indices distinguish repeated readouts from the same PMT.
        h_digit = hits.get("digit_idx")
        h_row = None
        if h_digit is not None:
            h_digit = np.asarray(h_digit, dtype=np.int64)
            if (
                h_digit.size == h_sensor.size
                and h_digit.size > 0
                and 0 <= int(h_digit.min())
                and int(h_digit.max()) < N
                and np.array_equal(sensor_idx[h_digit], h_sensor)
            ):
                h_row = h_digit
        if h_row is None:
            row_of_sensor = {int(s): i for i, s in enumerate(sensor_idx)}
            keep_hit = np.array([s in row_of_sensor for s in h_sensor], dtype=bool)
            h_row = np.array(
                [row_of_sensor.get(int(s), -1) for s in h_sensor], dtype=np.int64
            )
        else:
            keep_hit = np.ones(h_sensor.shape[0], dtype=bool)

        truth = self._particle_truth(src["labl_reader"], event_idx)

        part_hitcount = {}
        for p in np.unique(h_part):
            cnt = np.unique(h_sensor[keep_hit & (h_part == p)]).size
            if cnt >= self.min_ring_hits and int(p) in truth:
                part_hitcount[int(p)] = cnt

        ring_particles = sorted(part_hitcount, key=part_hitcount.get, reverse=True)
        if len(ring_particles) > self.max_rings:
            ring_particles = ring_particles[: self.max_rings]
        ring_particles_set = set(ring_particles)
        part_to_ring = {p: r for r, p in enumerate(ring_particles)}
        R = self.max_rings

        inst_pe = np.zeros((N, R), dtype=np.float32)
        valid_rows = (
            keep_hit & np.isin(h_part, list(ring_particles_set))
            if ring_particles
            else keep_hit & False
        )
        ring_col = np.array(
            [part_to_ring[int(p)] for p in h_part[valid_rows]], dtype=np.int64
        )
        np.add.at(inst_pe, (h_row[valid_rows], ring_col), h_pe[valid_rows])

        ring_pid = np.full(R, -1, dtype=np.int64)
        ring_anc_pdg = np.zeros(R, dtype=np.int64)  # earliest-ancestor (primary) pdg
        ring_anc_id = np.full(R, -1, dtype=np.int64)  # ... and its track id
        ring_momentum = np.zeros(R, dtype=np.float32)
        ring_vertex_valid = np.zeros(R, dtype=bool)
        ring_vertex = np.zeros((R, 3), dtype=np.float32)
        ring_direction = np.zeros((R, 3), dtype=np.float32)
        ring_valid = np.zeros(R, dtype=bool)
        for p in ring_particles:
            r = part_to_ring[p]
            t = truth[p]
            mass = _PDG_MASS_MEV.get(int(t["pdg"]), 0.0)
            ke = t["ke"] if np.isfinite(t["ke"]) else 0.0
            e_tot = ke + mass
            mom = float(np.sqrt(max(e_tot * e_tot - mass * mass, 0.0)))
            mom_eff = max(mom, 1.0)
            ring_pid[r] = pdg_to_pid(t["pdg"])
            ring_anc_pdg[r] = int(t.get("anc_pdg", t["pdg"]))
            ring_anc_id[r] = t["anc_id"]
            ring_momentum[r] = np.log(mom_eff)
            if t["vertex"] is not None:
                ring_vertex[r] = t["vertex"]
                ring_vertex_valid[r] = True
            if t["direction"] is not None:
                ring_direction[r] = t["direction"]
            ring_valid[r] = True

        out = dict(
            coord=coord,
            energy=energy,
            time=time,
            inst_pe=inst_pe,
            ring_pid=ring_pid,
            ring_anc_pdg=ring_anc_pdg,
            ring_anc_id=ring_anc_id,
            ring_momentum=ring_momentum,
            ring_vertex=ring_vertex,
            ring_vertex_valid=ring_vertex_valid,
            ring_direction=ring_direction,
            ring_valid=ring_valid,
            name=f"{src['name']}/{sample.get('name', event_idx)}",
        )
        return out

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


RING_KEYS = (
    "ring_pid",
    "ring_anc_pdg",
    "ring_anc_id",
    "ring_valid",
    "ring_momentum",
    "ring_direction",
    "ring_vertex",
    "ring_vertex_valid",
)
T0_HITS, T0_WINDOW = (
    8,
    10.0,
)  # first light: the earliest site time with >= 8 site times within 10 ns


def first_light(time):
    """The earliest time followed by at least T0_HITS - 1 more within T0_WINDOW ns (the earliest if there is none)."""
    t = np.sort(np.asarray(time).reshape(-1))
    if len(t) < T0_HITS:
        return t[0]
    ok = np.flatnonzero(t[T0_HITS - 1 :] - t[: len(t) - T0_HITS + 1] <= T0_WINDOW)
    return t[ok[0]] if ok.size else t[0]


def rotate_event(event, rot, coord_scale, grid_size):
    """A PMT-level event rotated by ``rot`` (3, 3) about the table's centre: hit positions, rings' vertices and
    directions turn with it, and the sites are rebuilt from the rotated positions (PMTs sharing a rotated voxel form
    one site), so every input stays on the encoder's grid."""
    coord = event["coord"] @ rot.T
    vox = np.floor(
        (coord / np.float32(coord_scale)).astype(np.float64) / grid_size
    ).astype(np.int64)
    out = dict(
        event,
        coord=coord.astype(np.float32),
        site_key=GridSample.fnv_hash_vec(vox - vox.min(0)),
    )
    for key in ("ring_vertex", "ring_direction"):
        out[key] = event[key] @ rot.T
    return out


def random_rotation(axes):
    """A rotation by uniform angles about the axes drawn with their probabilities (``axes``: axis -> probability),
    applied z, x, y in turn as pimm's RandomRotate transforms are."""
    rot = np.eye(3, dtype=np.float32)
    for axis in ("z", "x", "y"):
        if axis in axes and np.random.rand() < axes[axis]:
            c, s = np.cos(a := np.random.uniform(-np.pi, np.pi)), np.sin(a)
            i, j = {"z": (0, 1), "x": (1, 2), "y": (0, 2)}[axis]
            step = np.eye(3, dtype=np.float32)
            step[i, i], step[i, j], step[j, i], step[j, j] = c, -s, s, c
            rot = step @ rot
    return rot


def random_mirror(p):
    """Mirror each of x, y and z independently with probability ``p`` (pimm's RandomFlip, as a matrix)."""
    return np.diag([-1.0 if np.random.rand() < p else 1.0 for _ in range(3)]).astype(
        np.float32
    )


def site_times(event):
    """Each site's earliest time (sites in key order) and each PMT's site."""
    keys, site = np.unique(event["site_key"], return_inverse=True)
    time = np.full(len(keys), np.inf, np.float32)
    np.minimum.at(time, site, event["time"])
    return time, site


def overlay(a, b):
    """Two PMT-level events as one: each event's times from its own first light, then summed PE, the earlier time, and
    b's rings after a's (a's ring slots past its own hold b's; the caller keeps the total within the slots).
    """
    na, ra, rb = len(a["pmt"]), int(a["ring_valid"].sum()), int(b["ring_valid"].sum())
    pmt, first, where = np.unique(
        np.r_[a["pmt"], b["pmt"]], return_index=True, return_inverse=True
    )
    ia, ib = where[:na], where[na:]
    energy = np.zeros(len(pmt), np.float32)
    energy[ia] = a["energy"]
    energy[ib] += b["energy"]
    time = np.full(len(pmt), np.inf, np.float32)
    time[ia] = a["time"] - first_light(site_times(a)[0])
    time[ib] = np.minimum(time[ib], b["time"] - first_light(site_times(b)[0]))
    inst_pe = np.zeros((len(pmt), a["inst_pe"].shape[1]), np.float32)
    inst_pe[ia] = a["inst_pe"]
    inst_pe[ib, ra : ra + rb] += b["inst_pe"][:, :rb]
    out = dict(
        a,
        pmt=pmt,
        energy=energy,
        time=time,
        inst_pe=inst_pe,
        name=f"{a['name']}_overlay2",
        coord=np.r_[a["coord"], b["coord"]][first],
        site_key=np.r_[a["site_key"], b["site_key"]][first],
    )
    for key in RING_KEYS:
        out[key] = a[key].copy()
        out[key][ra : ra + rb] = b[key][:rb]
    return out


def build_sites(event, coord_scale, grid_size):
    """PMT-level event -> its hit sites: normalized position of the drawn PMT, the integer grid, summed PE
    and per-ring PE, and time from first light."""
    time, site = site_times(event)
    n = len(time)
    energy = np.zeros(n, np.float32)
    np.add.at(energy, site, event["energy"])
    inst_pe = np.zeros((n, event["inst_pe"].shape[1]), np.float32)
    np.add.at(inst_pe, site, event["inst_pe"])
    # one hit PMT per site gives its position: a site's PMTs in id order, one drawn per site by the event's seed
    order = np.argsort(site, kind="stable")
    count = np.bincount(site, minlength=n)
    pick = order[
        np.cumsum(count)
        - count
        + np.random.default_rng(event["seed"]).integers(0, count)
    ]
    coord = event["coord"][pick] / np.float32(coord_scale)
    grid = np.floor(coord / np.float32(grid_size)).astype(
        np.int64
    )  # retain the float32 voxel grid
    out = dict(
        coord=coord,
        grid_coord=grid - grid.min(0),
        energy=energy[:, None],
        time=(time - first_light(time))[:, None],
        inst_pe=inst_pe,
        hit=np.ones((n, 1), np.float32),
        hidden=np.zeros((n, 1), np.float32),
        name=event["name"],
    )
    out.update({key: event[key] for key in RING_KEYS})
    out["ring_vertex"] = event["ring_vertex"] / np.float32(
        coord_scale
    )  # the sites' normalized frame
    return out


PACK = 100  # files per pack


class _Events:
    """Digits, particle hits and labl truth for one split. HDF5 files open lazily per
    process, the most recent ``max_open`` kept open. A packed copy of the split, next to it as
    ``<config>_packed``, is read instead when present: the same events in ``PACK`` files per pack, so few files open.
    """

    def __init__(self, base, split, events, max_open=256):
        self.base, self.split, self.events, self.max_open = (
            base,
            split,
            events,
            max_open,
        )
        self.packed = os.path.isdir(f"{base}_packed")
        self.files, self.pid = OrderedDict(), None
        with h5py.File(self.path("sensor", int(events[0][0])), "r", locking=False) as f:
            self.pmt = f["config/sensor_positions"][:].astype(np.float32)

    @staticmethod
    def open(path):
        """A file with its metadata cache held at 1 MB: HDF5's default cache keeps growing over the events read (about
        0.6 GB per 1,000 across the 126 packs a worker keeps open); capped, a worker stays near 1 GB.
        """
        handle = h5py.File(path, "r", locking=False)
        cache = handle.id.get_mdc_config()
        cache.set_initial_size, cache.initial_size, cache.min_size, cache.max_size = (
            True,
            2**20,
            2**20,
            2**20,
        )
        handle.id.set_mdc_config(cache)
        return handle

    def path(self, modality, file):
        if self.packed:
            return os.path.join(
                f"{self.base}_packed", modality, f"pack_{file // PACK:03d}.h5"
            )
        return os.path.join(self.base, modality, f"wc_{modality}_{file:04d}.h5")

    def group(self, modality, e):
        if self.pid != os.getpid():  # every dataloader worker opens its own handles
            self.files, self.pid = OrderedDict(), os.getpid()
        file, number = (int(x) for x in self.events[e])
        key = (modality, file // PACK if self.packed else file)
        handle = (
            self.files.pop(key)
            if key in self.files
            else self.open(self.path(modality, file))
        )
        self.files[key] = handle  # most recently used last
        if len(self.files) > self.max_open:
            self.files.popitem(last=False)[1].close()
        return handle[
            (
                f"wc_{file:04d}/event_{number:03d}"
                if self.packed
                else f"event_{number:03d}"
            )
        ]

    def __getitem__(self, e):
        sensor, hits = self.group("sensor", e), self.group("hits", e)
        pmt = sensor["sensor_idx"][:].astype(np.int64)
        file, number = (int(x) for x in self.events[e])
        return dict(
            name=f"{self.split}/wc_sensor_{file:04d}/event_{number:03d}",
            sensor=dict(
                coord=self.pmt[pmt],
                energy=sensor["PE"][:].astype(np.float32),
                time=sensor["T"][:].astype(np.float32),
                sensor_idx=pmt,
            ),
            hits={
                key: hits[name][:]
                for key, name in (
                    ("sensor_idx", "sensor_idx"),
                    ("particle_idx", "particle_idx"),
                    ("energy", "PE"),
                    ("digit_idx", "digit_idx"),
                )
            },
        )

    def read_event(self, e):
        labl = self.group("labl", e)
        return {
            f"labl_{scope}_{name}": labl[f"per_{scope}"][name][:]
            for scope in ("particle", "track", "interaction")
            for name in labl[f"per_{scope}"]
        }


@DATASETS.register_module()
class WANDRingSitesDataset(PILArNetOverlayMixin, _RingTruth):
    """WAND-SK hit sites with overlapping ring truth (module docstring).

    Each sample (before transforms): ``coord`` (N, 3, normalized), ``grid_coord`` (N, 3), ``energy`` (N, 1, PE),
    ``time`` (N, 1, ns from first light), ``hit`` / ``hidden`` (N, 1), ``inst_pe`` (N, max_rings), ``ring_pid``,
    ``ring_anc_pdg``, ``ring_anc_id``, ``ring_valid``, ``ring_momentum``, ``ring_vertex_valid`` (max_rings,),
    ``ring_direction``, ``ring_vertex`` (max_rings, 3) and ``name``.

    Args:
        data_root (str): a WAND_SK_v4-layout dataset.
        split (str): ``"train"`` (the encoder's pool), ``"val"`` or ``"test"``. Defaults to
            ``"train"``.
        all_train_events (bool): ``"train"`` lists every event of the split's files, not only the encoder's
            > 1024-PMT pool, in a seed-0 shuffled order. Defaults to ``False``.
        config (str): Defaults to ``"config_01"``.
        first_event (int): skip the split's first ``first_event`` events. Defaults to ``0``.
        max_len (int): keep the first ``max_len`` events after that (-1: all). Defaults to ``-1``.
        subset_seed (int | None): ``"train"`` shuffles with this seed instead of 0, so ``max_len`` keeps a
            different draw. Defaults to ``None`` (seed 0).
        max_rings (int): ring slots per event (and the mixing cap). Defaults to ``32``.
        min_ring_hits (int): PMTs a particle must light to be a ring. Defaults to ``3``.
        coord_scale (float): metres per normalized unit. Defaults to ``18.1``.
        grid_size (float): site voxel, normalized units. Defaults to ``0.03`` (``--wand_grid``).
        overlay_prob (float): probability of mixing in a second event. Defaults to ``0.0``.
        random_rotation (dict): rotate each training sample (after mixing) by uniform angles about these axes with
            these probabilities, e.g. ``dict(z=0.8, x=0.5, y=0.5)``, rebuilding its sites (training only). Defaults to
            ``None``.
        random_flip (float): mirror each training sample (after mixing) across x, y and z, each with this probability,
            rebuilding its sites (training only). Defaults to ``0.0``.
        transform (list[dict]): transform configs. Defaults to ``None``.
        loop (int): epoch multiplier. Defaults to ``1``.
    """

    def __init__(
        self,
        data_root,
        split="train",
        config="config_01",
        first_event=0,
        max_len=-1,
        max_rings=32,
        min_ring_hits=3,
        coord_scale=18.1,
        grid_size=0.03,
        overlay_prob=0.0,
        random_rotation=None,
        random_flip=0.0,
        all_train_events=False,
        transform=None,
        loop=1,
        test_mode=False,
        subset_seed=None,
    ):
        if test_mode:
            raise NotImplementedError(
                "WANDRingSitesDataset scores whole events; there is no fragmented test mode"
            )
        base = os.path.join(data_root, split, config)
        assert subset_seed is None or split == "train", "subset_seed draws train events"
        order = 0 if subset_seed is None else int(subset_seed)
        if split == "train" and not all_train_events:
            rows = np.load(
                os.path.join(data_root, "metadata", "subset_train_over1024.npy")
            )
            seeds = np.random.default_rng(order).permutation(
                len(rows)
            )  # fixed event order
            events = rows[seeds]
        elif (
            split == "train"
        ):  # every train event: file NNNN holds its index rows as event_0, ...
            index = np.load(os.path.join(data_root, "metadata", "event_index.npz"))
            files = np.sort(index["file_index"][index["split_code"] == 0])
            events = np.stack(
                [files, np.arange(len(files)) - np.searchsorted(files, files)], 1
            )
            seeds = np.random.default_rng(order).permutation(len(events))
            events = events[seeds]
        elif split in ("val", "test"):
            events = []
            for name in sorted(os.listdir(os.path.join(base, "sensor"))):
                with h5py.File(os.path.join(base, "sensor", name), "r", locking=False) as f:
                    numbers = sorted(int(k[6:]) for k in f if k.startswith("event_"))
                events += [(int(name[10:14]), number) for number in numbers]
            events = np.array(events, np.int64)
            seeds = np.arange(len(events))
        else:
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
        events, seeds = events[first_event:], seeds[first_event:]
        if max_len > 0:
            events, seeds = events[:max_len], seeds[:max_len]
        self.reader, self.seeds = _Events(base, split, events), seeds
        self.sources = [dict(dataset=self.reader, labl_reader=self.reader, name=config)]
        self.data_list = [(0, e) for e in range(len(events))]
        self.split, self.min_points, self.max_rings, self.min_ring_hits = (
            split,
            1,
            int(max_rings),
            int(min_ring_hits),
        )
        self.coord_scale, self.grid_size = float(coord_scale), float(grid_size)
        self.overlay_n_events, self.overlay_prob = 2, float(overlay_prob)
        self.overlay_max_instances = dict(ring_valid=self.max_rings)
        # Hash each PMT voxel in the full table's lattice.
        vox = np.floor(
            (self.reader.pmt / np.float32(coord_scale)).astype(np.float64) / grid_size
        ).astype(np.int64)
        self.pmt_key = GridSample.fnv_hash_vec(vox - vox.min(0))
        self.transform, self.loop, self.test_mode = Compose(transform), int(loop), False
        self.random_rotation = dict(random_rotation) if random_rotation else None
        self.random_flip = float(random_flip)

    def get_data(self, idx):
        """The event's hit PMTs in id order (position, PE, time, site key, per-ring PE), its rings and its seed."""
        e = idx % len(self.data_list)
        digits = super().get_data(e)
        sensor = self.reader.group("sensor", e)["sensor_idx"][:].astype(np.int64)
        assert len(sensor) == len(digits["energy"]), "the join resampled an empty event"
        order = np.argsort(sensor, kind="stable")
        pmt, start = np.unique(sensor[order], return_index=True)
        event = dict(
            pmt=pmt,
            coord=self.reader.pmt[pmt],
            site_key=self.pmt_key[pmt],
            seed=int(self.seeds[e]),
            energy=np.add.reduceat(digits["energy"].reshape(-1)[order], start),
            time=np.minimum.reduceat(digits["time"].reshape(-1)[order], start),
            inst_pe=np.add.reduceat(digits["inst_pe"][order], start, axis=0),
        )
        event.update({key: digits[key] for key in RING_KEYS + ("name",)})
        return event

    @staticmethod
    def _instance_counts(event, keys):
        """Rings per event, for the mixin's donor cap."""
        return {key: int(event[key].sum()) for key in keys}

    def _apply_overlay(self, event):
        counts = self._instance_counts(event, ("ring_valid",))
        donor, _ = self._draw_donor(
            len(self.data_list), counts, ("ring_valid",), event["name"]
        )
        return event if donor is None else overlay(event, donor)

    def prepare_train_data(self, idx):
        event = self._maybe_overlay(self.get_data(idx))
        if (
            self.random_rotation or self.random_flip
        ):  # after mixing, which overlays the two events PMT by PMT
            rot = random_rotation(self.random_rotation or {})
            if self.random_flip:
                rot = (
                    random_mirror(self.random_flip) @ rot
                )  # rotations, then flips, as pimm's transforms run
            event = rotate_event(event, rot, self.coord_scale, self.grid_size)
        return self.transform(build_sites(event, self.coord_scale, self.grid_size))


PROBE_SPECIES = {
    "e": 11,
    "mu": 13,
    "pi": 211,
    "pi0": 111,
    "gamma": 22,
}  # |PDG|, charges merged


def primary_truth(index, rows):
    """The single primary's vertex (m), unit initial direction and |p| (MeV/c, from its kinetic energy and PDG mass) of
    each ``metadata/event_index.npz`` row."""
    first = index["primary_offsets"][rows]
    ke = index["primary_ke_MeV"][first].astype(np.float64)
    mass = np.array([_PDG_MASS_MEV[abs(int(p))] for p in index["primary_pdg"][first]])
    direction = index["primary_dir"][first].astype(np.float64)
    return dict(
        vertex=index["vertex"][rows].astype(np.float32),
        direction=(direction / np.linalg.norm(direction, axis=1, keepdims=True)).astype(
            np.float32
        ),
        momentum=np.sqrt(ke * (ke + 2 * mass)).astype(np.float32),
    )


@DATASETS.register_module()
class WANDProbeDataset(WANDRingSitesDataset):
    """The single-particle probe events of a split (pools built by ``pimm.engines.hooks.eval.probes.wc``),
    as the encoder's hit sites plus each event's truth and task membership.

    A task (``dict(name, kind, species)``) is ``pid`` (``species`` = (signal, background)) or ``vertex`` /
    ``direction`` / ``momentum`` (``species`` = (one,)). Train: a kinematic task holds the first ``n`` events of its
    species' pool, a PID task the first ``n // 2`` of each class; val: every pool event of the task's species. The
    events are the union, species by species in pool order. Extra sample keys (per event): ``task_mask`` (1, T),
    ``probe_label`` (1, T; 1 where the event is its PID task's signal), ``vertex`` (1, 3, m), ``direction`` (1, 3) and
    ``momentum`` (1,, MeV/c). No augmentation, no mixing.

    Args:
        pools (str): the pools ``.npz``.
        tasks (list[dict]): the probe tasks, in head order.
        split (str): ``"train"`` or ``"val"``. Defaults to ``"train"``.
        n (int): the label budget (train only). Defaults to ``1000``.
    """

    def __init__(
        self,
        data_root,
        pools,
        tasks,
        split="train",
        n=1000,
        coord_scale=18.1,
        grid_size=0.03,
        transform=None,
        loop=1,
    ):
        super().__init__(
            data_root,
            split=split,
            all_train_events=split == "train",
            coord_scale=coord_scale,
            grid_size=grid_size,
            transform=transform,
            loop=loop,
        )
        pools = np.load(pools)
        take = {}  # species -> events used, a prefix of its pool
        for task in tasks:
            budget = n if task["kind"] != "pid" else n // 2
            for s in task["species"]:
                take[s] = max(
                    take.get(s, 0),
                    budget if split == "train" else len(pools[f"{split}_{s}"]),
                )
        positions, rows, species, rank = [], [], [], []
        for s, k in take.items():
            assert k <= len(
                pools[f"{split}_{s}"]
            ), f"{split} {s}: {k} events asked, the pool has fewer"
            positions.append(pools[f"{split}_{s}"][:k])
            rows.append(pools[f"{split}_{s}_row"][:k])
            species.append(np.full(k, s))
            rank.append(np.arange(k))  # place in its species' pool
        positions, rows, species, rank = (
            np.concatenate(a) for a in (positions, rows, species, rank)
        )
        self.task_mask = np.zeros((len(positions), len(tasks)), bool)
        self.probe_label = np.zeros((len(positions), len(tasks)), np.float32)
        for t, task in enumerate(tasks):
            budget = n if task["kind"] != "pid" else n // 2
            self.task_mask[:, t] = np.isin(species, task["species"]) & (
                (rank < budget) if split == "train" else True
            )
            self.probe_label[:, t] = (
                (species == task["species"][0]) if task["kind"] == "pid" else 0
            )
        index = np.load(os.path.join(data_root, "metadata", "event_index.npz"))
        assert (
            index["hit_pmts"][rows] > 0
        ).all(), "an event without hits would be resampled"
        self.truth = primary_truth(index, rows)
        self.reader.events, self.seeds = (
            self.reader.events[positions],
            self.seeds[positions],
        )
        self.data_list = [(0, e) for e in range(len(positions))]

    def prepare_train_data(self, idx):
        e = idx % len(self.data_list)
        sites = build_sites(self.get_data(idx), self.coord_scale, self.grid_size)
        sites.update(
            task_mask=self.task_mask[e][None],
            probe_label=self.probe_label[e][None],
            **{key: value[e][None] for key, value in self.truth.items()},
        )
        return self.transform(sites)

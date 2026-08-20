"""Shared PILArNet v3/v3_extra event decoding.

``decode_event`` turns one event's raw HDF5/parquet arrays into the flat,
point-aligned dictionary consumed by the pimm dataset and transform pipeline.
The ``v3_extra`` schema preserves the v3 tables and adds particle momentum,
lineage, and an explicit per-interaction vertex table.
"""

from __future__ import annotations

import numpy as np

from pimm.utils.logger import get_root_logger

# Priority for voxel deduplication: track > shower > Michel > delta > LED.
DEFAULT_LABEL_PRIORITY = {1: 0, 0: 1, 2: 2, 3: 3, 4: 4}

CLUSTER_WIDTH = 6
CLUSTER_EXTRA_WIDTH = 6
CLUSTER_EXTRA_2_WIDTH = 5
INTERACTION_EXTRA_WIDTH = 4
INVALID = -1.0

# Extra point-aligned truth emitted by the extended decoder. Configs can pass
# these to ``default_transform(extra_keys=...)`` when the model needs them.
V3_EXTRA_KEYS = (
    "mass",
    "px",
    "py",
    "pz",
    "momentum_vector",
    "momentum_vec",
    "true_energy",
    "parent_pdg",
    "parent_track_id",
    "interaction_vertex",
    "is_primary_4cls",
)


def resolve_revision(revision: str) -> str:
    """Normalize a configured revision to a supported v3-family layout."""
    if revision in ("v3", "v3_extra"):
        return revision
    if revision == "v2":
        get_root_logger().warning(
            "PILArNet revision='v2' requested; reading v3 instead."
        )
        return "v3"
    raise ValueError(
        f"PILArNet revision={revision!r} is not supported; "
        "choose 'v3' or 'v3_extra'."
    )


def _leading_cluster_index(cluster_id, group_id, cluster_size):
    """Return each particle group's leading-cluster row for every cluster.

    EM showers can contain several clusters under one particle ``group_id``.
    Per-particle truth must come from the group's primary cluster
    (``cluster_id == group_id``), not whichever shower fragment happens to be
    selected after point aggregation. If that cluster is absent, use the
    group's largest cluster.
    """
    cluster_id = np.asarray(cluster_id, dtype=np.int64)
    group_id = np.asarray(group_id, dtype=np.int64)
    cluster_size = np.asarray(cluster_size, dtype=np.int64)
    lead_row: dict[int, int] = {}
    for i, group in enumerate(group_id):
        group = int(group)
        current = lead_row.get(group)
        if current is None:
            lead_row[group] = i
            continue
        is_primary = cluster_id[i] == group
        current_is_primary = cluster_id[current] == group
        if is_primary and not current_is_primary:
            lead_row[group] = i
        elif is_primary == current_is_primary and cluster_size[i] > cluster_size[current]:
            lead_row[group] = i
    return np.asarray([lead_row[int(group)] for group in group_id], dtype=np.int64)


def _true_energy(momentum, mass):
    """Compute ``sqrt(|p|^2 + m^2)`` in GeV, preserving invalid sentinels."""
    momentum = np.asarray(momentum, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    valid = (momentum >= 0) & (mass >= 0)
    energy = np.sqrt(np.clip(momentum * momentum + mass * mass, 0.0, None))
    return np.where(valid, energy, INVALID).astype(np.float32)


def _interaction_vertices(interaction_id, interaction_extra, fallback):
    """Broadcast an explicit ``[id, x, y, z]`` table to cluster rows."""
    if interaction_extra is None:
        return np.asarray(fallback, dtype=np.float32)
    raw = np.asarray(interaction_extra)
    if raw.size == 0:
        return np.full((len(interaction_id), 3), INVALID, dtype=np.float32)
    table = raw.reshape(-1, INTERACTION_EXTRA_WIDTH)
    lookup = {int(row[0]): row[1:4] for row in table if row[0] >= 0}
    return np.asarray(
        [lookup.get(int(value), (INVALID, INVALID, INVALID)) for value in interaction_id],
        dtype=np.float32,
    )


def decode_event(
    point,
    cluster,
    cluster_extra,
    *,
    revision: str = "v3",
    cluster_extra_2=None,
    interaction_extra=None,
    energy_threshold: float = 0.0,
    remove_low_energy_scatters: bool = False,
    old_pid_mapping: bool = False,
) -> dict:
    """Decode one PILArNet v3-family event into point-aligned arrays.

    The base v3 layout uses ``cluster_extra`` width 6:
    ``[mass_MeV, |p|_GeV, vtx_x, vtx_y, vtx_z, is_primary]``.
    ``v3_extra`` additionally stores a row-aligned ``cluster_extra_2`` width 5
    table ``[px, py, pz, parent_pdg, parent_track_id]`` and an
    ``interaction_extra`` width 4 table ``[interaction_id, x, y, z]``. Packed
    width-9/11 ``cluster_extra`` variants are accepted for compatibility.

    Missing optional extended columns in base v3 produce ``-1`` sentinels, so
    callers can use one stable output schema across both revisions.
    """
    revision = resolve_revision(revision)
    data = np.asarray(point).reshape(-1, 8)[:, [0, 1, 2, 3]]

    cluster_arr = np.asarray(cluster).reshape(-1, CLUSTER_WIDTH)
    cluster_size, cluster_id, group_id, interaction_id, semantic_id, pid = (
        cluster_arr[:, [0, 1, 2, 3, 4, 5]].T
    )
    cluster_size = np.asarray(cluster_size, dtype=np.int64)
    n_clusters = cluster_arr.shape[0]

    raw_extra = np.asarray(cluster_extra) if cluster_extra is not None else None
    if raw_extra is None:
        raise ValueError("PILArNet v3 requires a `cluster_extra` table; got None.")
    extra = (
        raw_extra.reshape(n_clusters, -1)
        if n_clusters > 0
        else np.empty((0, CLUSTER_EXTRA_WIDTH), dtype=np.float32)
    )
    if extra.shape[1] not in (6, 9, 11):
        raise ValueError(
            "Expected cluster_extra width 6, 9, or 11 for a v3-family shard; "
            f"got {extra.shape[1]}."
        )
    mass, momentum, vtx_x, vtx_y, vtx_z, is_primary = extra[:, :6].T

    px = np.full(n_clusters, INVALID, dtype=np.float32)
    py = np.full(n_clusters, INVALID, dtype=np.float32)
    pz = np.full(n_clusters, INVALID, dtype=np.float32)
    parent_pdg = np.full(n_clusters, INVALID, dtype=np.float32)
    parent_track_id = np.full(n_clusters, INVALID, dtype=np.float32)

    if cluster_extra_2 is not None:
        extra_2 = (
            np.asarray(cluster_extra_2).reshape(n_clusters, -1)
            if n_clusters > 0
            else np.empty((0, CLUSTER_EXTRA_2_WIDTH), dtype=np.float32)
        )
        if extra_2.shape[1] != CLUSTER_EXTRA_2_WIDTH:
            raise ValueError(
                f"Expected cluster_extra_2 width {CLUSTER_EXTRA_2_WIDTH}, "
                f"got {extra_2.shape[1]}."
            )
        px, py, pz, parent_pdg, parent_track_id = extra_2.T
    elif extra.shape[1] >= 9:
        px, py, pz = extra[:, 6:9].T
        if extra.shape[1] == 11:
            parent_pdg, parent_track_id = extra[:, 9:11].T
    elif revision == "v3_extra":
        raise ValueError(
            "revision='v3_extra' requires cluster_extra_2 (split layout) or "
            "packed cluster_extra width 9/11."
        )

    if revision == "v3_extra" and interaction_extra is None:
        raise ValueError("revision='v3_extra' requires an `interaction_extra` table.")

    # Rest mass is stored in MeV; all momentum quantities are GeV.
    mass = np.where(mass == INVALID, INVALID, mass / 1.0e3).astype(np.float32)
    pid = pid.copy()
    pid[pid == -1] = 6 if old_pid_mapping else 5

    # Use one consistent particle-level truth row across fragmented EM groups.
    if n_clusters:
        leading = _leading_cluster_index(cluster_id, group_id, cluster_size)
        momentum = momentum[leading]
        mass = mass[leading]
        px, py, pz = px[leading], py[leading], pz[leading]
        parent_pdg = parent_pdg[leading]
        parent_track_id = parent_track_id[leading]

    true_energy = _true_energy(momentum, mass)
    cluster_vertex = np.stack([vtx_x, vtx_y, vtx_z], axis=1).astype(np.float32)
    interaction_vertex = _interaction_vertices(
        interaction_id, interaction_extra, fallback=cluster_vertex
    )

    if remove_low_energy_scatters and n_clusters:
        data = data[cluster_size[0] :]
        cluster_size = cluster_size[1:]
        semantic_id, group_id, interaction_id, pid = (
            semantic_id[1:],
            group_id[1:],
            interaction_id[1:],
            pid[1:],
        )
        momentum, mass, px, py, pz, true_energy = (
            momentum[1:],
            mass[1:],
            px[1:],
            py[1:],
            pz[1:],
            true_energy[1:],
        )
        parent_pdg, parent_track_id = parent_pdg[1:], parent_track_id[1:]
        cluster_vertex = cluster_vertex[1:]
        interaction_vertex = interaction_vertex[1:]
        is_primary = is_primary[1:]

    def repeat(values):
        return np.repeat(values, cluster_size, axis=0)

    data_semantic_id = repeat(semantic_id)
    data_group_id = repeat(group_id)
    data_interaction_id = repeat(interaction_id)
    data_pid = repeat(pid)
    data_momentum = repeat(momentum)
    data_mass = repeat(mass)
    data_px, data_py, data_pz = repeat(px), repeat(py), repeat(pz)
    data_true_energy = repeat(true_energy)
    data_parent_pdg = repeat(parent_pdg)
    data_parent_track_id = repeat(parent_track_id)
    data_vertex = repeat(cluster_vertex)
    data_interaction_vertex = repeat(interaction_vertex)
    data_is_primary = repeat(is_primary)

    if len(data) != len(data_semantic_id):
        raise ValueError(
            "PILArNet cluster sizes do not match the point table: "
            f"sum(cluster_size)={len(data_semantic_id)}, points={len(data)}."
        )

    if energy_threshold > 0:
        keep = data[:, 3] > energy_threshold
        data = data[keep]
        data_semantic_id = data_semantic_id[keep]
        data_group_id = data_group_id[keep]
        data_interaction_id = data_interaction_id[keep]
        data_pid = data_pid[keep]
        data_momentum = data_momentum[keep]
        data_mass = data_mass[keep]
        data_px, data_py, data_pz = data_px[keep], data_py[keep], data_pz[keep]
        data_true_energy = data_true_energy[keep]
        data_parent_pdg = data_parent_pdg[keep]
        data_parent_track_id = data_parent_track_id[keep]
        data_vertex = data_vertex[keep]
        data_interaction_vertex = data_interaction_vertex[keep]
        data_is_primary = data_is_primary[keep]

    momentum_vector = np.stack([data_px, data_py, data_pz], axis=1).astype(np.float32)
    momentum_vec = momentum_vector.copy()
    valid_direction = ~(momentum_vec == INVALID).all(axis=1)
    if valid_direction.any():
        norm = np.linalg.norm(momentum_vec[valid_direction], axis=1, keepdims=True)
        momentum_vec[valid_direction] /= np.clip(norm, 1.0e-9, None)

    # 0 primary, 1 Michel, 2 delta, 3 other secondary.
    is_primary_4cls = np.where(data_is_primary == 1, 0, 3).astype(np.int32)
    is_primary_4cls[data_semantic_id == 2] = 1
    is_primary_4cls[data_semantic_id == 3] = 2

    particle_ids = data_group_id.astype(np.int32)
    interaction_ids = data_interaction_id.astype(np.int32)
    return {
        "coord": data[:, :3].astype(np.float32),
        "energy": data[:, 3].astype(np.float32)[:, None],
        "momentum": data_momentum.astype(np.float32)[:, None],
        "mass": data_mass.astype(np.float32)[:, None],
        "px": data_px.astype(np.float32)[:, None],
        "py": data_py.astype(np.float32)[:, None],
        "pz": data_pz.astype(np.float32)[:, None],
        "momentum_vector": momentum_vector,
        "momentum_vec": momentum_vec,
        "true_energy": data_true_energy.astype(np.float32)[:, None],
        "parent_pdg": data_parent_pdg.astype(np.int32)[:, None],
        "parent_track_id": data_parent_track_id.astype(np.int32)[:, None],
        "vertex": data_vertex.astype(np.float32),
        "interaction_vertex": data_interaction_vertex.astype(np.float32),
        "is_primary": data_is_primary.astype(np.int32)[:, None],
        "is_primary_4cls": is_primary_4cls[:, None],
        "segment_motif": data_semantic_id.astype(np.int32)[:, None],
        "segment_pid": data_pid.astype(np.int32)[:, None],
        "instance_particle": map_instance_ids(particle_ids),
        "instance_interaction": map_instance_ids(interaction_ids),
        "segment_interaction": (interaction_ids[:, None] != -1).astype(np.int32),
    }


def map_instance_ids(instance_ids_array):
    """Compact non-negative instance ids to ``0..K-1`` and preserve ``-1``."""
    unique_ids = np.unique(instance_ids_array)
    mapping = {
        old_id: new_id
        for new_id, old_id in enumerate(unique_ids[unique_ids >= 0])
    }
    return np.asarray(
        [mapping.get(value, -1) for value in instance_ids_array], dtype=np.int32
    )[:, None]

"""NEXT's two-blob topological classifier (Simón et al. 2021, arXiv 2102.11931 Sec. 5): hits re-binned into
5 mm voxels, voxels touching by a face, edge or corner form a track, and the blobs are spheres around the end points of
the longest track; the weaker blob's energy is the signal score.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# Adapted from next-exp/IC @3e8881e, reco/paolina_functions.py and evm/event_model.py (MIT).

CORNER = 1.8  # IC types.symbols.Contiguity.CORNER


class Voxel:
    def __init__(self, x, y, z, E, size, hits):
        self.xyz, self.E, self.size, self.hits = (
            (x, y, z),
            E,
            size,
            hits,
        )

    @property
    def pos(self):
        return np.array(self.xyz)


class Track:
    def __init__(self, voxels, graph):
        self.voxels = voxels
        order = sorted(range(len(voxels)), key=lambda i: voxels[i].xyz)
        distances = dijkstra(graph, directed=False)
        self.distances = {
            voxels[i]: {voxels[j]: distances[i, j] for j in order} for i in order
        }


def voxelize_hits(hits, voxel_dimensions):
    xyz = hits["X Y Z".split()].values
    hlo, hhi = xyz.min(axis=0), xyz.max(axis=0)
    bounding_box_centre = (hhi + hlo) / 2
    bounding_box_size = hhi - hlo

    number_of_voxels = np.ceil(bounding_box_size / voxel_dimensions).astype(int)
    number_of_voxels = np.clip(number_of_voxels, a_min=1, a_max=None)

    half_range = bounding_box_size / 2

    voxel_edges_lo = bounding_box_centre - half_range
    voxel_edges_hi = bounding_box_centre + half_range
    eps = 3e-12
    voxel_edges_lo -= eps
    voxel_edges_hi += eps

    E, edges = np.histogramdd(
        xyz,
        bins=number_of_voxels,
        range=tuple(zip(voxel_edges_lo, voxel_edges_hi)),
        weights=hits["E"],
    )

    def centres(a):
        return (a[1:] + a[:-1]) / 2

    def sizes(a):
        return a[1:] - a[:-1]

    cx, cy, cz = map(centres, edges)
    size_x, size_y, size_z = map(sizes, edges)
    hits_indices = np.array(
        [
            np.clip(np.digitize(coords, bins, right=False) - 1, 0, n - 1)
            for coords, bins, n in zip(xyz.T, edges, number_of_voxels)
        ]
    ).T

    voxels = []
    true_dimensions = np.array([size_x[0], size_y[0], size_z[0]])
    for indices_voxel in np.stack(np.nonzero(E)).T:
        x, y, z = indices_voxel
        indices = np.where(np.all(hits_indices == indices_voxel, axis=1))[0]
        voxels.append(
            Voxel(cx[x], cy[y], cz[z], E[x, y, z], true_dimensions, hits.iloc[indices])
        )
    return voxels


def make_track_graphs(voxels):
    rows, cols, weights = [], [], []
    for i, j in combinations(range(len(voxels)), 2):
        va, vb = voxels[i], voxels[j]
        delta = va.pos - vb.pos
        if np.linalg.norm(delta / va.size) < CORNER:
            rows.extend((i, j))
            cols.extend((j, i))
            weights.extend((np.linalg.norm(delta),) * 2)
    graph = csr_matrix((weights, (rows, cols)), shape=(len(voxels), len(voxels)))
    count, labels = connected_components(graph, directed=False)
    return tuple(
        Track(tuple(voxels[i] for i in indices), graph[indices][:, indices])
        for label in range(count)
        for indices in (np.flatnonzero(labels == label),)
    )


def find_extrema_and_length(distance):
    if len(distance) == 1:
        only_voxel = next(iter(distance))
        return (only_voxel, only_voxel, 0.0)
    first, last, max_distance = None, None, 0
    for (voxel1, dist_from_voxel_1_to), (voxel2, _) in combinations(
        distance.items(), 2
    ):
        d = dist_from_voxel_1_to[voxel2]
        if d > max_distance:
            first, last, max_distance = voxel1, voxel2, d
    return first, last, max_distance


def length(track):
    return find_extrema_and_length(track.distances)[2]


def energy_of_voxels_within_radius(distances, radius):
    return sum(v.E for (v, d) in distances.items() if d < radius)


def blob_centre(voxel):
    positions = voxel.hits["X Y Z".split()]
    energies = voxel.hits["E"]
    if sum(energies):
        return np.average(positions, weights=energies, axis=0)
    return voxel.pos


def hits_in_blob(track_graph, radius, extreme):
    distances = track_graph.distances
    dist_from_extreme = distances[extreme]
    blob_pos = blob_centre(extreme)
    diag = np.linalg.norm(extreme.size)

    blob_hits = []
    for v in track_graph.voxels:
        voxel_distance = dist_from_extreme[v]
        if voxel_distance < radius + diag:
            hit_distances = np.linalg.norm(v.hits["X Y Z".split()] - blob_pos, axis=1)
            blob_hits.append(v.hits.loc[hit_distances < radius])
    return pd.concat(blob_hits, ignore_index=True)


def blob_energies(track_graph, small_radius):
    """IC's blob_energies with big_radius=None: (blob1, blob2) energies, the larger first."""
    distances = track_graph.distances
    a, b, _ = find_extrema_and_length(distances)
    ha = hits_in_blob(track_graph, small_radius, a)
    hb = hits_in_blob(track_graph, small_radius, b)
    some_hits = len(ha) or len(hb)
    Ea = (
        ha["E"].sum()
        if some_hits
        else energy_of_voxels_within_radius(distances[a], small_radius)
    )
    Eb = (
        hb["E"].sum()
        if some_hits
        else energy_of_voxels_within_radius(distances[b], small_radius)
    )
    return (Eb, Ea) if Eb > Ea else (Ea, Eb)


def blob2_energy(coord, energy, radii, voxel=5.0):
    """Blob-2 energy of the longest track at each radius in ``radii`` (mm); coord (N, 3) mm, energy (N,) MeV."""
    hits = pd.DataFrame(dict(X=coord[:, 0], Y=coord[:, 1], Z=coord[:, 2], E=energy))
    tracks = make_track_graphs(voxelize_hits(hits, np.full(3, voxel)))
    track = max(tracks, key=length)
    return np.array([blob_energies(track, r)[1] for r in radii])

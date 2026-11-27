"""Worker-side masks, view sites and nearest-neighbour matches for Panda V2 self-distillation."""

import numpy as np
import torch
from scipy.spatial import cKDTree

from .common import TRANSFORMS


@TRANSFORMS.register_module()
class SonataPreparation:
    """Place after Collect: panda-ssl-v2m1 trains only on batches prepared here.

    Each global view is divided into cubic patches of side ``mask_size``, and each patch gets a random priority from
    the worker's NumPy RNG. When a batch is collated, the ``mask_ratio`` fraction of its patches with the lowest
    priorities is masked; if every patch of a view is masked, the view's highest-priority patch stays visible. With
    ``observable``, masked rows are removed from the student's views; otherwise they are flagged for the mask token,
    and their coordinates are jittered by ``mask_jitter``. The rows of a view that share a voxel are merged into a
    site, and the sites are pooled with the encoder's functions and ``strides`` up to the level of the losses
    (``up_cast_level`` upcasts above the coarsest stage), where the nearest-neighbour matches are found. ``model`` is
    the panda-ssl-v2m1 config the model is built from; the model checks that the settings read from it are its own.
    """

    def __init__(self, model):
        from pimm.models.panda.pretrain.panda_ssl_v2m1 import self_distillation_settings

        settings = self_distillation_settings(model)
        self.settings = dict(
            strides=tuple(settings["backbone"]["strides"]),
            **{
                key: settings[key]
                for key in (
                    "observable",
                    "up_cast_level",
                    "mask_size",
                    "mask_ratio",
                    "mask_jitter",
                    "match_max_r",
                    "observable_match_max_r",
                )
            },
        )

    def __call__(self, data):
        coord = np.asarray(data["global_coord"])
        ends = np.asarray(data["global_offset"], dtype=np.int64)
        counts = np.diff(ends, prepend=0)
        if np.any(counts <= 0) or np.any(
            np.diff(np.asarray(data["local_offset"]), prepend=0) <= 0
        ):
            raise ValueError("SonataPreparation needs points in every global and local view")
        view = np.repeat(np.arange(len(ends)), counts)
        minimum = np.minimum.reduceat(coord, np.r_[0, ends[:-1]], axis=0)
        grid = np.floor_divide(
            coord - minimum[view], coord.dtype.type(self.settings["mask_size"])
        ).astype(np.int64)
        patches, patch = np.unique(
            np.column_stack((view, grid)), axis=0, return_inverse=True
        )
        scores = np.random.random(len(patches))
        jitter, noise = self.settings["mask_jitter"], None
        if not self.settings["observable"] and jitter > 0:
            # Sonata's jitter: Gaussian, clipped from above at twice its scale
            noise = np.minimum(
                np.random.standard_normal(coord.shape) * jitter, 2 * jitter
            ).astype(coord.dtype)
        data["sonata_plan"] = SamplePlan(
            self.settings, patch.reshape(-1), patches[:, 0], scores, noise
        )
        return data


class SamplePlan:
    """One sample's mask patches, with a collate hook that prepares the complete batch."""

    def __init__(self, settings, patch, views, scores, noise):
        self.settings, self.patch, self.views = settings, patch, views
        self.scores, self.noise = scores, noise

    @staticmethod
    def collate(samples, batch):
        settings = samples[0].settings
        if any(sample.settings != settings for sample in samples[1:]):
            raise ValueError("SonataPreparation settings must agree across a batch")
        grid_size = float(batch["grid_size"].reshape(-1)[0])
        if not bool((batch["grid_size"] == grid_size).all()):
            raise ValueError("SonataPreparation needs one grid size per batch")
        global_views = len(batch["global_offset"]) // len(samples)
        local_views = len(batch["local_offset"]) // len(samples)
        settings = dict(
            settings, num_global_view=global_views, num_local_view=local_views
        )

        scores = np.concatenate([sample.scores for sample in samples])
        starts = np.cumsum([0] + [len(sample.scores) for sample in samples[:-1]])
        patch = np.concatenate(
            [sample.patch + start for sample, start in zip(samples, starts)]
        )
        views = np.concatenate(
            [sample.views + i * global_views for i, sample in enumerate(samples)]
        )
        masked = np.zeros(len(scores), dtype=bool)
        masked[np.argsort(scores)[: int(len(scores) * settings["mask_ratio"])]] = True
        by_priority = np.lexsort((scores, views))
        best = by_priority[  # each view's highest-priority patch
            np.r_[np.flatnonzero(np.diff(views[by_priority])), len(views) - 1]
        ]
        masked[best[np.bincount(views[~masked], minlength=len(best)) == 0]] = False
        row_masked = torch.from_numpy(masked[patch])

        coord, origin = batch["global_coord"], batch["global_origin_coord"]
        teacher, teacher_level = _view_sites(
            coord, origin, batch["global_offset"], grid_size, settings
        )
        plan = dict(settings=settings)
        if settings["observable"]:
            visible = (~row_masked).nonzero().flatten()
            ends = torch.bincount(
                teacher["row_batch"][visible], minlength=len(teacher["row_offset"])
            ).cumsum(0)
            student, student_level = _view_sites(
                coord[visible], origin[visible], ends, grid_size, settings
            )
            plan["visible"] = visible
        else:
            if samples[0].noise is None:
                student, student_level = teacher, teacher_level
            else:
                noise = torch.from_numpy(np.concatenate([sample.noise for sample in samples]))
                coord = torch.where(row_masked[:, None], coord + noise, coord)
                student, student_level = _view_sites(
                    coord, origin, batch["global_offset"], grid_size, settings
                )
            plan["mask"], plan["coord"] = row_masked, coord
        local, local_level = _view_sites(
            batch["local_coord"],
            batch["local_origin_coord"],
            batch["local_offset"],
            grid_size,
            settings,
        )

        same = np.arange(len(teacher_level[1]))
        rolled = (
            same // global_views * global_views + global_views - 1 - same % global_views
        )
        radius = settings["observable_match_max_r"]
        if not settings["observable"] and (
            student is teacher or settings["up_cast_level"] == 3
        ):
            # the student and teacher rows (or sites) are the same, in the same order
            mask_match = torch.arange(len(teacher_level[0]))[:, None].repeat(1, 2)
        else:
            mask_match = _match_views(*student_level, *teacher_level, same, radius)
        plan.update(
            sites=dict(teacher=teacher, student=student, local=local),
            mask_match=mask_match,
            roll_mask_match=_match_views(*student_level, *teacher_level, rolled, radius),
            local_match=_match_views(
                *local_level,
                *teacher_level,
                np.arange(len(local_level[1])) // local_views * global_views,
                settings["match_max_r"],
            ),
        )
        return plan


def _view_sites(coord, origin, offset, grid_size, settings):
    """Merge the rows of each view by voxel and pool the sites as the encoder does, up to the level of the losses.

    Returns the sites, and the origins and view ends at that level: the input rows with their sites' mean origin for
    ``up_cast_level=3``, otherwise the sites pooled ``3 - up_cast_level`` times.
    """
    import torch_scatter

    from pimm.models.panda.encoder.panda_encoder_v2m1 import pool

    coord = np.asarray(coord)
    ends = torch.as_tensor(offset).long()
    row_batch = torch.repeat_interleave(
        torch.arange(len(ends)), torch.diff(ends, prepend=ends.new_zeros(1))
    )
    grid = np.trunc((coord - coord.min(0)) / coord.dtype.type(grid_size)).astype(np.int64)
    grid, batch, index = pool(torch.from_numpy(grid), row_batch, 1)
    sites = dict(
        index=index,
        grid_coord=grid,
        batch=batch,
        offset=torch.bincount(batch, minlength=len(ends)).cumsum(0),
        row_batch=row_batch,
        row_offset=ends,
    )
    # each site's mean origin, its rows summed in row order as in the encoder's pooling
    order = torch.argsort(index, stable=True)
    count = torch.bincount(index, minlength=len(grid))
    ptr = torch.cat([count.new_zeros(1), torch.cumsum(count, 0)])
    origin = torch_scatter.segment_csr(torch.as_tensor(origin).double().index_select(0, order), ptr, reduce="sum")
    origin = origin / count[:, None].double()
    if settings["up_cast_level"] == 3:
        return sites, (origin[index].numpy(), ends.numpy())
    for stride in settings["strides"][: 3 - settings["up_cast_level"]]:
        grid, batch, cluster = pool(grid, batch, stride)
        order = torch.argsort(cluster, stable=True)
        count = torch.bincount(cluster, minlength=len(grid))
        ptr = torch.cat([count.new_zeros(1), torch.cumsum(count, 0)])
        origin = torch_scatter.segment_csr(origin.index_select(0, order), ptr, reduce="sum") / count[:, None].double()
    return sites, (
        origin.numpy(),
        torch.bincount(batch, minlength=len(ends)).cumsum(0).numpy(),
    )


def _match_views(source, source_ends, target, target_ends, view_map, radius):
    """Nearest target rows within each selected view, with the same strict radius cut."""
    matches = []
    source_starts, target_starts = (
        np.r_[0, source_ends[:-1]],
        np.r_[0, target_ends[:-1]],
    )
    first_views = np.r_[0, np.flatnonzero(np.diff(view_map)) + 1]
    last_views = np.r_[first_views[1:] - 1, len(view_map) - 1]
    for first, last in zip(first_views, last_views):
        target_view = view_map[first]
        begin, end = int(source_starts[first]), int(source_ends[last])
        target_begin, target_end = int(target_starts[target_view]), int(
            target_ends[target_view]
        )
        if begin == end or target_begin == target_end:
            continue
        distance, neighbour = cKDTree(target[target_begin:target_end]).query(
            source[begin:end], k=1, workers=1
        )
        valid = distance < radius
        matches.append(
            np.column_stack(
                (np.arange(begin, end)[valid], neighbour[valid] + target_begin)
            )
        )
    array = np.concatenate(matches) if matches else np.empty((0, 2), dtype=np.int64)
    return torch.from_numpy(array.astype(np.int64, copy=False))

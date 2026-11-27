import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Sequence

from pimm.models.losses import build_criteria
from pimm.models.utils.structure import Point
from pimm.models.utils import offset2batch, offset2bincount
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    """Point-wise segmentor: a backbone and a linear or MLP head on ``segment``. With ``heads``
    (``{name: dict(num_classes=..., criteria=[...], label=name, mlp=mlp_head)}``), several heads on the same
    features instead, each predicting ``input_dict[label]`` with its own loss, summed; ``seg_logits`` is then a
    dict keyed by head name. On a frozen backbone the heads train independently (e.g. a linear and an MLP probe
    in one run).

    The loss is computed on the voxels. In evaluation, the output is given at the original points: ``seg_logits``,
    ``offset`` (the batch's ``origin_offset``) and, when the batch is labelled, each label under its own name (from
    ``origin_<label>``). The batch needs ``inverse`` from ``GridSample(return_inverse=True)``."""

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        mlp_head=False,
        mlp_hidden_channels=None,
        heads=None,
    ):
        super().__init__()

        def head(classes, mlp=mlp_head):
            if not mlp:
                return (
                    nn.Linear(backbone_out_channels, classes)
                    if classes > 0
                    else nn.Identity()
                )
            hidden_channels = (
                backbone_out_channels // 2
                if mlp_hidden_channels is None
                else int(mlp_hidden_channels)
            )
            if hidden_channels <= 0:
                raise ValueError("mlp_hidden_channels must be greater than zero")
            return nn.Sequential(
                nn.Linear(backbone_out_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(hidden_channels, classes),
            )

        self.heads = dict(heads or {})
        if self.heads:
            self.seg_head = nn.ModuleDict(
                {
                    key: head(spec["num_classes"], spec.get("mlp", mlp_head))
                    for key, spec in self.heads.items()
                }
            )
            self.head_criteria = {
                key: build_criteria(spec["criteria"])
                for key, spec in self.heads.items()
            }
            self.head_labels = {
                key: spec.get("label", key) for key, spec in self.heads.items()
            }
        else:
            self.seg_head = head(num_classes)
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode=True):
        """Set module mode while keeping a frozen probe backbone in eval mode.

        ``nn.Module.train()`` recurses into every child module.  The trainer calls
        it at the start of each epoch, so setting the backbone to eval mode only
        in ``__init__`` is not enough: BatchNorm buffers would keep updating and
        dropout/drop-path would remain active during a nominal linear probe.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        labels = list(dict.fromkeys(self.head_labels.values())) if self.heads else ["segment"]
        labelled = self.training or all(label in input_dict for label in labels)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        if self.heads:
            seg_logits = {key: self.seg_head[key](feat) for key in self.heads}
            if labelled:
                return_dict["loss"] = sum(
                    self.head_criteria[key](
                        seg_logits[key], input_dict[label].view(-1).long()
                    )
                    for key, label in self.head_labels.items()
                )
        else:
            seg_logits = self.seg_head(feat)
            if labelled:
                return_dict["loss"] = self.criteria(seg_logits, input_dict["segment"])
        if self.training:
            return return_dict
        if "inverse" not in input_dict:
            raise KeyError(
                "DefaultSegmentorV2 predicts the original points in evaluation: add "
                "GridSample(return_inverse=True) to the evaluation transforms and collect "
                "'inverse', with offset_keys_dict=dict(offset='coord', origin_offset='inverse')"
            )
        # inverse gives each original point's voxel within its event
        offset, origin_offset = input_dict["offset"], input_dict["origin_offset"]
        voxel = input_dict["inverse"] + (offset - offset2bincount(offset)).repeat_interleave(
            offset2bincount(origin_offset)
        )
        return_dict["seg_logits"] = (
            {key: logits[voxel] for key, logits in seg_logits.items()}
            if self.heads
            else seg_logits[voxel]
        )
        return_dict["offset"] = origin_offset
        if labelled:
            return_dict.update({label: input_dict[f"origin_{label}"] for label in labels})
        return return_dict


@MODELS.register_module()
class DefaultSegmentorV3(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = point.pred_logits

        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultInsSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        imprint_weights=False,

    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["query_truth"])
            return_dict["loss"] = loss
        # eval
        elif "query_truth" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["query_truth"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        import torch_cluster

        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        import torch_scatter

        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)

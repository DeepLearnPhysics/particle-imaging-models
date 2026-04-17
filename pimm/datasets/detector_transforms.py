"""
Detector-specific transforms for multimodal detector data.

PDGToSemantic: derives semantic labels from PDG codes in seg data (3D tasks).
    Used when no label file is available (fallback).
    For production training, labels come from the labl file via
    JAXTPCDataset._apply_labl_to_3d() or _build_corr_pointcloud().
"""

import numpy as np
from .transform import TRANSFORMS


@TRANSFORMS.register_module()
class PDGToSemantic:
    """Fallback: derive approximate semantic labels from PDG codes.

    Use this only when no labl file is available. For production training,
    use modalities=('seg', 'labl') which applies labels via JAXTPCLablReader.

    Schemes
    -------
    motif_5cls : shower(0), track(1), michel(2), delta(3), led(4)
    pid_6cls : photon(0), electron(1), muon(2), pion(3), proton(4), other(5)
    custom : user-provided {pdg_code: class_index} dict
    """

    _MOTIF = {
        22: 0, 11: 0, -11: 0,              # shower
        13: 1, -13: 1,                      # track (muon)
        211: 1, -211: 1,                    # track (pion)
        2212: 1,                             # track (proton)
        321: 1, -321: 1,                    # track (kaon)
    }

    _PID = {
        22: 0,                               # photon
        11: 1, -11: 1,                      # electron
        13: 2, -13: 2,                      # muon
        211: 3, -211: 3,                    # pion
        2212: 4,                             # proton
    }

    def __init__(self, scheme='motif_5cls', custom_map=None):
        self.scheme = scheme
        if scheme == 'motif_5cls':
            self.mapping = self._MOTIF
            self.default = 4
        elif scheme == 'pid_6cls':
            self.mapping = self._PID
            self.default = 5
        elif scheme == 'custom':
            assert custom_map is not None
            self.mapping = custom_map
            self.default = max(custom_map.values()) + 1
        elif scheme == 'none':
            self.mapping = None
            self.default = -1
        else:
            raise ValueError(f"Unknown label scheme: {scheme}")

    def __call__(self, data_dict):
        if self.mapping is None or 'pdg' not in data_dict:
            return data_dict

        # Skip if labels already loaded from label file
        if 'segment' in data_dict or 'segment_motif' in data_dict:
            return data_dict

        pdg = data_dict['pdg']
        n = len(pdg)
        labels = np.full(n, self.default, dtype=np.int32)
        for code, cls in self.mapping.items():
            labels[pdg == code] = cls

        data_dict['segment_motif'] = labels[:, None]

        # Also produce PID labels
        if self.scheme == 'motif_5cls':
            pid = np.full(n, 5, dtype=np.int32)
            for code, cls in self._PID.items():
                pid[pdg == code] = cls
            data_dict['segment_pid'] = pid[:, None]
        elif self.scheme == 'pid_6cls':
            data_dict['segment_pid'] = labels[:, None]

        # Derive instance from track_ids (simple contiguous remapping)
        if 'instance_particle' not in data_dict and 'track_ids' in data_dict:
            track_ids = data_dict['track_ids']
            mask = track_ids >= 0
            if mask.any():
                _, inverse = np.unique(track_ids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict['instance_particle'] = out[:, None]
            else:
                data_dict['instance_particle'] = np.full((n, 1), -1, dtype=np.int32)

        if 'instance_interaction' not in data_dict and 'interaction_ids' in data_dict:
            iids = data_dict['interaction_ids']
            mask = iids >= 0
            if mask.any():
                _, inverse = np.unique(iids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict['instance_interaction'] = out[:, None]
            else:
                data_dict['instance_interaction'] = np.full((n, 1), -1, dtype=np.int32)

            data_dict['segment_interaction'] = (iids[:, None] != -1).astype(np.int32)

        return data_dict

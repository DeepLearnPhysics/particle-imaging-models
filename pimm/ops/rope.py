"""Rotary position tables and application over three coordinate axes."""

import torch


# Rotary embeddings from RoFormer (Su et al., 2021).
def rope_tables(pos, head_dim, base=100.0):
    """cos/sin for 3D rotary embeddings: each third of the head dim is rotated by one grid axis."""
    d = head_dim // 3
    angle = pos[:, None, :, None].float() * base ** (
        -torch.arange(0, d, 2, device=pos.device) / d
    )
    angle = torch.cat([angle, angle], -1)
    return angle.cos(), angle.sin()


def apply_rope(x, cos, sin):
    x = x.unflatten(-1, (3, -1))
    x1, x2 = x.chunk(2, -1)
    return (x * cos + torch.cat([-x2, x1], -1) * sin).flatten(-2)

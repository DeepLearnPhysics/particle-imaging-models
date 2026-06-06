"""Front-end helpers for z-order and Hilbert point serialization."""

import torch
import numpy as np
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode_int as hilbert_encode_
from .hilbert import decode as hilbert_decode_

@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    """Encode integer grid coordinates into one serialization code per point."""
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def encode_batch(grid_coord, batch=None, depth=16, orders=("z",)):
    """
    Encode multiple serialization orders in parallel.
    
    Instead of calling encode() sequentially for each order, this batches
    the coordinate transformations and processes them together.
    
    Returns: (num_orders, num_points) tensor of codes
    """
    n_points = grid_coord.shape[0]
    device = grid_coord.device
    
    # separate orders by encoding type
    z_orders = []
    hilbert_orders = []
    order_indices = {}  # maps order name to output index
    
    for i, order in enumerate(orders):
        order_indices[order] = i
        if order in ("z", "z-trans"):
            z_orders.append(order)
        elif order in ("hilbert", "hilbert-trans"):
            hilbert_orders.append(order)
    
    # pre-allocate output
    codes = torch.empty((len(orders), n_points), dtype=torch.int64, device=device)
    
    # batch z-order encoding (both normal and transposed together)
    if z_orders:
        z_coords = []
        z_indices = []
        for order in z_orders:
            if order == "z":
                z_coords.append(grid_coord)
            else:  # z-trans
                z_coords.append(grid_coord[:, [1, 0, 2]])
            z_indices.append(order_indices[order])
        
        # stack and encode all z-order variants at once
        stacked = torch.cat(z_coords, dim=0)  # (num_z_orders * n_points, 3)
        x = stacked[:, 0].long()
        y = stacked[:, 1].long()
        z = stacked[:, 2].long()
        all_z_codes = z_order_encode_(x, y, z, b=None, depth=depth)
        
        # split back and assign
        z_code_chunks = all_z_codes.split(n_points)
        for idx, out_idx in zip(range(len(z_orders)), z_indices):
            codes[out_idx] = z_code_chunks[idx]
    
    # batch hilbert encoding (both normal and transposed together)
    if hilbert_orders:
        h_coords = []
        h_indices = []
        for order in hilbert_orders:
            if order == "hilbert":
                h_coords.append(grid_coord)
            else:  # hilbert-trans
                h_coords.append(grid_coord[:, [1, 0, 2]])
            h_indices.append(order_indices[order])
        
        # stack and encode all hilbert variants at once
        stacked = torch.cat(h_coords, dim=0)  # (num_hilbert_orders * n_points, 3)
        all_h_codes = hilbert_encode_(stacked, num_dims=3, num_bits=depth)
        
        # split back and assign
        h_code_chunks = all_h_codes.split(n_points)
        for idx, out_idx in zip(range(len(hilbert_orders)), h_indices):
            codes[out_idx] = h_code_chunks[idx]
    
    # add batch info if provided
    if batch is not None:
        batch = batch.long()
        codes = (batch.unsqueeze(0) << (depth * 3)) | codes
    
    return codes


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    """Decode serialized point codes into grid coordinates and batch ids."""
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    """Encode ``(N, 3)`` grid coordinates with Morton z-order keys."""
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    """Decode Morton z-order keys into ``(N, 3)`` grid coordinates."""
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    """Encode ``(N, 3)`` grid coordinates with Hilbert curve keys."""
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    """Decode Hilbert curve keys into ``(N, 3)`` grid coordinates."""
    return hilbert_decode_(code, num_dims=3, num_bits=depth)

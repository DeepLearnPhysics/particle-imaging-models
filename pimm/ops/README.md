# pimm.ops

This submodule contains operation primitives that have been accelerated to speed up training or reduce memory use.
Each function below, except `hilbert_encode` and `set_attention_backend`, can be run with `fused=True` or `fused=False`. With `fused=True`, the
accelerated version of the operation is run. With `fused=False`, the original PyTorch or spconv version of the code is run.

If an input is outside what the accelerated version supports (for example, attention dropout, which is not implemented in the masked attention kernel), a `RuntimeError` is raised. The inputs, outputs and limits of each function are documented in its docstring.

## `varlen_attention`

```python
from pimm.ops import varlen_attention

out = varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       softmax_scale=None, dropout_p=0.0, attn_mask=None, fused=None)
```

Softmax attention within packed variable-length sequences, with the same API as `flash_attn_varlen_func` in
[FlashAttention](https://github.com/Dao-AILab/flash-attention) ([Dao, 2023](https://arxiv.org/abs/2307.08691)). `q` is `[total_q, heads, dim]`, `k` and `v` are `[total_k, heads, dim]`, and the rows of sequence `b`
are `cu_seqlens[b]:cu_seqlens[b + 1]` (int32). `max_seqlen_*` are upper bounds on the sequence lengths. `attn_mask` is an optional
additive mask, given as each sequence's `[len_q, len_k]` block, row-major, concatenated over the sequences. The output
has the layout of `q`.

- `fused=True` without a mask: FlashAttention is run (`flash-attn` on Ampere or newer,
[flash-attention-turing](https://github.com/youngsm/flash-attention-turing) on Turing).
- `fused=True` with a mask: a [Triton](https://triton-lang.org) kernel is run. The mask blocks are read directly from the jagged layout, so no
sequence is padded, and the softmax is accumulated in FP32 with FlashAttention's tiled online softmax
([Dao et al., 2022](https://arxiv.org/abs/2205.14135)). No gradient is computed for the mask.
- `fused=False`: the sequences are padded into one batch for PyTorch's
[`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
- Without `fused`: `fused=True` is used with the `"flash"` attention backend (the default) and `fused=False` with
`"torch"`; see [`set_attention_backend`](#set_attention_backend).

## `serialized_attention`

```python
from pimm.ops import serialized_attention
from pimm.ops.rope import rope_tables

cos, sin = rope_tables(grid_coord[gather], head_dim)
out = serialized_attention(qkv,gather, scatter, cos, sin, query_offsets, offsets, max_seqlen, heads, fused=True)
```

Attention within fixed-size windows of serialized points, as in Point Transformer V3
([Wu et al., 2024](https://arxiv.org/abs/2312.10035)), with rotary position embeddings
([Su et al., 2021](https://arxiv.org/abs/2104.09864)) of the 3D grid coordinates. `qkv` is `[N, 3 * C]`. `gather`
is the point index of every window slot; a point can appear in two windows, and `scatter` is the slot of each
point's primary copy. `offsets` and `query_offsets` are the boundaries of each window's slots and of its primary
queries. `cos` and `sin` are the rotary tables of the gathered rows. The output is `[N, C]`, in point order and in the dtype of `qkv`. The queries and
keys are rotated in FP32 and the attention is computed in FP16.

- `fused=True`: on Ampere or newer, the windows are gathered, rotated and packed in one Triton pass, and FlashAttention
is run on the primary queries only. On Turing, FlashAttention is run through flash-attention-turing.
- `fused=False`: PyTorch's attention is run, on any CUDA GPU.
- Without `fused`: `fused=True` is used with the `"flash"` attention backend (the default) and `fused=False` with
`"torch"`; see [`set_attention_backend`](#set_attention_backend).

## `set_attention_backend`

```python
from pimm.ops import set_attention_backend

set_attention_backend("torch")  # or "flash", the default
```

Call `set_attention_backend` to choose, for the whole process, the attention of `varlen_attention` and
`serialized_attention` calls without `fused`: with `"flash"`, `fused=True` is used, and with `"torch"`, `fused=False`.
`fused` is left out in the Panda V2 models, so you choose their attention for each run rather than when you build a
model. In training and test runs, the backend is set from the config key `attention_backend`, so you can pass
`attention_backend=torch` after `--` on the command line; with `pimm.from_pretrained`, pass `attention_backend="torch"`.
If FlashAttention can't run on your GPU, an error that names `attention_backend="torch"` is raised, and PyTorch's
attention is not used in its place.

## `sinkhorn_distillation_loss`

```python
from pimm.ops import sinkhorn_distillation_loss

loss = sinkhorn_distillation_loss(student_features, student_prototypes,
                                  teacher_features, teacher_prototypes,
                                  student_temperature, teacher_temperature,
                                  iterations=3, marginal=None, fused=False, distributed=True)  # [N]
```

The per-row distillation loss of Sonata ([Wu et al., 2025](https://arxiv.org/abs/2503.16429)). The teacher's scores against `[K, D]` prototypes are balanced with `iterations`
Sinkhorn steps, and the cross-entropy of the student's scores against the balanced assignments is returned for each of
the `N` matched rows. The teacher inputs are detached. `marginal` replaces the uniform prototype distribution with a
`[K]` target. With `distributed=True`, the balancing spans every rank of the default process group.

- `fused=True`: Triton kernels are run with BF16 operands and FP32 accumulation, and like in Flash Attention 2, no `[N, K]` score table is stored.
Only first derivatives are supported, and the values differ from the PyTorch version by rounding.
- `fused=False`: the dense Sinkhorn and cross-entropy are computed in PyTorch.

## `SubMConv3d`

```python
from pimm.ops import SubMConv3d

conv = SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key="stage0", fused=True)
```

The submanifold sparse convolution `SubMConv3d` of [spconv](https://github.com/traveller59/spconv) (see
[spconv's usage guide](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md)), with the same arguments, parameters and checkpoint keys, so checkpoints are interchangeable
between the two. `fused` applies to training calls only; spconv is always used in evaluation.

In spconv, one of two algorithms is selected from the kernel size, and each is accelerated differently with
`fused=True`:

- For kernels with at most 32 entries (`kx * ky * kz <= 32`, so 3×3×3 or smaller for a cubic kernel), spconv's
implicit GEMM is used, in which each point's neighbours are stored as one 32-bit mask. With `fused=True`, the convolution  
maps are built without spconv's GPU<->CPU synchronization, and the convolution itself is run by spconv. An `indice_key`
and FP32 inputs without autocast are required. In the Panda V2 encoder, for example, these are the 3×3×3
convolutions of the first two stages.
- For larger kernels, spconv's Native algorithm is used. With `fused=True`, the convolution is computed by a Triton FP16 kernel,  
which is supported only for four input channels and odd kernel sizes. FP16 inputs or autocast are required. In the  
Panda V2 encoder, for example, this is the 5×5×5 embedding module, for detectors with four input channels.

With `fused=False`, spconv is ran unchanged.

## `nearest_neighbor`

```python
from pimm.ops import nearest_neighbor

index, distance = nearest_neighbor(query, query_offset, points, point_offset, fused=True)  # [M] int64, [M] FP32
```

The nearest point of the same event for each of `M` queries, used to match the views of an event in Sonata pretraining
([Wu et al., 2025](https://arxiv.org/abs/2503.16429)). `query` and `points` are `[M, 3]` and `[N, 3]` coordinates of
packed events, and `query_offset` and `point_offset` are the cumulative row counts of the events, as in a pimm `Point`.
The index is into `points`, and the distance is Euclidean, not squared, as in the `knn_query` of
[pointops](https://github.com/Pointcept/Pointcept/tree/main/libs/pointops), which was used before. Distances are
computed in FP32 from the coordinate differences, and ties go to the lowest index.
Queries of an event without points get index -1 and distance inf. No gradient is computed.

- `fused=True`: a Triton kernel is run on CUDA tensors. A block of queries of one event is assigned to each program,
the points of that event are scanned in tiles, and a running minimum and its index are kept, so no distance table is
stored. Multiply-adds are fused, as in pointops' CUDA build; on LArTPC and sPHENIX pretraining batches, the indices and
distances match pointops' bit for bit.
- `fused=False`: the distances of each event are computed in PyTorch, in chunks of queries, on any device. Because
each operation is rounded separately, the two versions can pick different points when two points are equally near
within rounding.

## `hilbert_encode`

```python
from pimm.ops import hilbert_encode

codes = hilbert_encode(grid_coord, depth)  # [N] int64
```

Hilbert-curve codes of `[N, 3]` non-negative integer coordinates below `2**depth`, used to serialize points as in
Point Transformer V3 ([Wu et al., 2024](https://arxiv.org/abs/2312.10035)), computed by a Triton kernel on a CUDA
device. The codes are identical, bit for bit, to those of the PyTorch encoder in `pimm.models.utils.serialization`,
which is used for CPU tensors. This function has no `fused` flag.

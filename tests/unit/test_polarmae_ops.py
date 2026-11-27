import numpy as np
import pytest
import torch

from pimm.models.polarmae import ops

try:
    from pytorch3d_ops import _C as p3d_C
    from pytorch3d_ops.loss import chamfer_distance as p3d_chamfer
    from pytorch3d_ops.ops import ball_query as p3d_ball_query
    from pytorch3d_ops.ops import knn_points as p3d_knn_points
except ImportError:
    p3d_C = None
try:
    from cnms import _ext as cnms_ext
except ImportError:
    cnms_ext = None

needs_p3d = pytest.mark.skipif(p3d_C is None, reason="pytorch3d_ops is not installed")
needs_cnms = pytest.mark.skipif(cnms_ext is None or p3d_C is None, reason="cnms is not installed")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, needs_cuda])]

# An empty event, a single point, short rows and a full row.
LENGTHS = [0, 1, 3, 40, 64]


def _batch(lattice, device, D=3, seed=0, lengths=LENGTHS, P=64):
    """Random padded events; with `lattice`, on small integers, which are summed exactly and
    tie often."""
    g = torch.Generator().manual_seed(seed)
    shape = (len(lengths), P, D)
    pts = torch.randint(0, 6, shape, generator=g).float() if lattice else torch.rand(shape, generator=g)
    return pts.to(device), torch.tensor(lengths, device=device)


def _valid(lengths, P):
    return torch.arange(P, device=lengths.device) < lengths[:, None]


@needs_p3d
@pytest.mark.parametrize("lattice", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_knn_matches_pytorch3d(device, lattice):
    p1, l1 = _batch(lattice, device, seed=0)
    p2, l2 = _batch(lattice, device, seed=1, lengths=[5, 0, 2, 64, 9])
    K = 4
    dists, idx = ops.knn_points(p1, p2, lengths1=l1, lengths2=l2, K=K)
    ref = p3d_knn_points(p1, p2, lengths1=l1, lengths2=l2, K=K, return_sorted=True)
    found = _valid(l1, p1.shape[1])[..., None] & (torch.arange(K, device=device) < l2[:, None, None])
    torch.testing.assert_close(dists[found], ref.dists[found], atol=1e-6, rtol=1e-5)
    if not lattice:  # ties may pick another point at the same distance
        assert torch.equal(idx[found], ref.idx[found])
    # The indices hold the distances returned, ties or not.
    nearest = torch.gather(p2, 1, idx.flatten(1)[..., None].expand(-1, -1, 3)).view(*idx.shape, 3)
    torch.testing.assert_close(((p1[:, :, None] - nearest) ** 2).sum(-1)[found], dists[found])
    # A short row of p2 pads with inf and index 0, a padded row of p1 with distance 0 and index 0.
    short = _valid(l1, p1.shape[1])[..., None] & ~found
    assert torch.isinf(dists[short]).all() and (idx[short] == 0).all()
    padded = ~_valid(l1, p1.shape[1])
    assert (idx[padded] == 0).all() and (dists[padded] == 0).all()


@needs_p3d
@pytest.mark.parametrize("K", [8, 64])
@pytest.mark.parametrize("lattice", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_ball_query_matches_pytorch3d(device, lattice, K):
    p1, l1 = _batch(lattice, device, seed=0)
    p2, l2 = _batch(lattice, device, seed=1, lengths=[5, 0, 2, 64, 9])
    # On the lattice, points at offsets (2, 0, 0) sit exactly on the radius and stay out.
    radius = 2.0 if lattice else 0.3
    idx = ops.ball_query(p1, p2, K=K, radius=radius, lengths1=l1, lengths2=l2)
    _, ref, _ = p3d_ball_query(p1, p2, K=K, radius=radius, lengths1=l1, lengths2=l2, return_nn=False)
    assert torch.equal(idx, ref)


@needs_p3d
@pytest.mark.parametrize("K", [16, 64])
@pytest.mark.parametrize("lattice", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_fps_matches_pytorch3d(device, lattice, K):
    pts, lengths = _batch(lattice, device)
    start = (torch.rand(len(LENGTHS), generator=torch.Generator().manual_seed(2)) * torch.tensor(LENGTHS)).long()
    start = start.to(device)
    idx = ops.sample_farthest_points(pts, lengths, K, start_idxs=start)
    K_t = torch.full_like(lengths, K)
    ref = p3d_C.sample_farthest_points(pts.contiguous(), lengths, K_t, start, K)
    # pytorch3d writes the start index even into an empty event; here that row is all -1.
    assert torch.equal(idx[lengths > 0], ref[lengths > 0])
    assert (idx[lengths == 0] == -1).all()


def _compiled_cnms_retained(centroids, radius, overlap_factor, K, lengths):
    """The compiled kernel's retained centroids, visited in the stable candidate order."""
    q = 2.0 * radius * overlap_factor
    _, idx, _ = p3d_ball_query(centroids, centroids, K=K, radius=q, lengths1=lengths, lengths2=lengths, return_nn=False)
    _, order = idx.ne(-1).sum(-1).sort(dim=-1, descending=True, stable=True)
    retain = cnms_ext.greedy_reduction_padded(order, idx, lengths, -1)
    return [centroids[b][retain[b]] for b in range(len(centroids))]


@needs_cnms
@pytest.mark.parametrize("lattice", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_cnms_matches_compiled_greedy(device, lattice):
    # Untruncated lists (K = P, as PoLAr-MAE calls it). With truncated, asymmetric lists the
    # compiled kernel also un-retains earlier retained centres that sit in a later one's list.
    pts, lengths = _batch(lattice, device)
    K = pts.shape[1]
    radius, overlap = (1.5, 0.75) if lattice else (0.25, 0.72)
    centroids, count = ops.cnms(pts, radius=radius, overlap_factor=overlap, K=K, lengths=lengths)
    ref = _compiled_cnms_retained(pts, radius, overlap, K, lengths)
    assert count.tolist() == [len(r) for r in ref]
    for b, r in enumerate(ref):
        assert torch.equal(centroids[b, : count[b]], r)


def _sequential_greedy(order, idx, lengths):
    B, P, _ = idx.shape
    retain = torch.zeros(B, P, dtype=torch.bool)
    for b in range(B):
        suppressed = set()
        for c in order[b, : lengths[b]].tolist():
            if c not in suppressed:
                retain[b, c] = True
                suppressed.update(j for j in idx[b, c].tolist() if j >= 0)
    return retain


def _random_neighbour_lists(seed, B=4, P=50, K=7):
    """Asymmetric, truncated neighbour lists with -1 padding, and a random visit order."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, P, (B, P, K), generator=g)
    idx[torch.rand(B, P, K, generator=g) < 0.3] = -1
    idx[:, torch.arange(P), 0] = torch.arange(P)  # every candidate holds itself
    lengths = torch.tensor([P, 0, 17, 1])[:B]
    order = torch.stack([torch.randperm(P, generator=g) for _ in range(B)])
    # Valid candidates come first in the order, as the descending counts put them.
    order = torch.stack([torch.cat([o[o < n], o[o >= n]]) for o, n in zip(order, lengths.tolist())])
    return order, idx, lengths


@pytest.mark.parametrize("seed", range(4))
def test_greedy_rounds_match_sequential(seed):
    order, idx, lengths = _random_neighbour_lists(seed)
    assert torch.equal(ops._greedy_rounds(order, idx, lengths), _sequential_greedy(order, idx, lengths))


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("seed", range(4))
def test_triton_greedy_matches_rounds(seed):
    from pimm.ops.triton.cnms import greedy_retain

    order, idx, lengths = _random_neighbour_lists(seed)
    expected = ops._greedy_rounds(order, idx, lengths)
    assert torch.equal(greedy_retain(order.cuda(), idx.cuda(), lengths.cuda()).cpu(), expected)
    pts, lengths = _batch(True, "cuda", seed=seed)
    on_gpu, n_gpu = ops.cnms(pts, radius=1.5, overlap_factor=0.75, K=64, lengths=lengths)
    on_cpu, n_cpu = ops.cnms(pts.cpu(), radius=1.5, overlap_factor=0.75, K=64, lengths=lengths.cpu())
    assert torch.equal(n_gpu.cpu(), n_cpu) and torch.equal(on_gpu.cpu(), on_cpu)


@needs_p3d
@pytest.mark.parametrize("device", DEVICES)
def test_chamfer_matches_pytorch3d(device):
    lengths = [32, 0, 5, 17]
    x, lx = _batch(False, device, D=4, seed=0, lengths=lengths, P=32)
    y, ly = _batch(False, device, D=4, seed=1, lengths=lengths, P=32)
    x.requires_grad_(True)
    loss = ops.chamfer_distance(x, y, lx, ly)
    (grad,) = torch.autograd.grad(loss, x)
    x_ref = x.detach().clone().requires_grad_(True)
    ref, _ = p3d_chamfer(x_ref, y, x_lengths=lx, y_lengths=ly)
    (grad_ref,) = torch.autograd.grad(ref, x_ref)
    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(grad, grad_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.gpu
@needs_cuda
def test_queries_ignore_autocast():
    p1, l1 = _batch(False, "cuda", seed=0)
    p2, l2 = _batch(False, "cuda", seed=1)
    expected = ops.knn_points(p1, p2, l1, l2, K=4), ops.ball_query(p1, p2, 8, 0.3, l1, l2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = ops.knn_points(p1, p2, l1, l2, K=4), ops.ball_query(p1, p2, 8, 0.3, l1, l2)
    assert torch.equal(got[0][0], expected[0][0]) and torch.equal(got[0][1], expected[0][1])
    assert torch.equal(got[1], expected[1])


def test_tokenizer_and_multiview_cnms_run_on_cpu():
    from pimm.datasets.transform.multiview import MultiViewGenerator
    from pimm.models.polarmae.layers import PointcloudGrouping

    pts, lengths = _batch(False, "cpu", D=4, lengths=[64, 40, 3])
    grouping = PointcloudGrouping(
        num_groups=32, context_length=16, group_max_points=8, group_radius=0.2,
        group_upscale_points=16, overlap_factor=0.72,
    )
    out = grouping(pts, lengths)
    assert out["groups"].shape == (3, 16, 8, 4) and out["embedding_mask"][:, 0].all()

    coord = np.random.default_rng(0).random((200, 3)).astype("f4")
    view = MultiViewGenerator(center_sampling="cnms", center_sampling_kwargs=dict(radius=0.1, overlap_factor=0.72, K=32))
    center = view.get_center(coord)
    assert (coord == center).all(1).any()


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("lattice", [False, True])
def test_triton_kernels_match_pytorch(lattice):
    """The Triton kernels on CUDA against the PyTorch code on CPU, on the same events. The points are
    the first three of four channels, so the kernels read strided rows."""
    q1, l1 = _batch(lattice, "cpu", seed=3, D=4)
    q2, l2 = _batch(lattice, "cpu", seed=4, D=4, lengths=[5, 0, 2, 64, 9])
    p1, p2, g1, g2 = q1[..., :3], q2[..., :3], q1.cuda()[..., :3], q2.cuda()[..., :3]
    gl1, gl2 = l1.cuda(), l2.cuda()
    radius = 2.0 if lattice else 0.3
    for K in (8, 64):
        assert torch.equal(ops.ball_query(g1, g2, K, radius, gl1, gl2).cpu(), ops.ball_query(p1, p2, K, radius, l1, l2))
    start = torch.tensor([0, 0, 2, 17, 63])
    for K in (16, 64):
        expected = ops.sample_farthest_points(p1, l1, K, start)
        assert torch.equal(ops.sample_farthest_points(g1, gl1, K, start.cuda()).cpu(), expected)
    for K in (1, 5):
        d_gpu, i_gpu = ops.knn_points(g1, g2, gl1, gl2, K)
        d_cpu, i_cpu = ops.knn_points(p1, p2, l1, l2, K)
        torch.testing.assert_close(d_gpu.cpu(), d_cpu, atol=1e-6, rtol=1e-5)
        if not lattice:  # on the lattice, torch.topk may return another of equally near points
            assert torch.equal(i_gpu.cpu(), i_cpu)
    with pytest.raises(ValueError):
        ops.knn_points(g1, g2, gl1, gl2, K=9)
    # Of equally near points, the kernel keeps the lowest indices, nearest first.
    pts = torch.zeros(1, 6, 3, device="cuda")
    pts[0, 3:, 0] = 1.0
    d, i = ops.knn_points(pts[:, :1], pts, K=4)
    assert i.tolist() == [[[0, 1, 2, 3]]] and d.tolist() == [[[0.0, 0.0, 0.0, 1.0]]]

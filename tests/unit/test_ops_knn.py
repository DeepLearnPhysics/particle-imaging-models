import pytest
import torch

from pimm.ops import nearest_neighbor


def _events(counts_q, counts_p, integer, seed=0, device="cpu"):
    generator = torch.Generator().manual_seed(seed)

    def coords(n):
        if integer:  # small integer coordinates are summed exactly and tie often
            return torch.randint(0, 4, (n, 3), generator=generator).float()
        return torch.randn(n, 3, generator=generator)

    query, points = coords(sum(counts_q)), coords(sum(counts_p))
    query_offset = torch.tensor(counts_q).cumsum(0)
    point_offset = torch.tensor(counts_p).cumsum(0)
    return (t.to(device) for t in (query, query_offset, points, point_offset))


def _brute_force(query, query_offset, points, point_offset):
    """The lowest index at the smallest squared distance, from the full distance table of each event."""
    index = torch.full((len(query),), -1)
    distance = torch.full((len(query),), float("inf"))
    q = [0] + query_offset.tolist()
    p = [0] + point_offset.tolist()
    for b in range(len(q) - 1):
        if p[b + 1] == p[b]:
            continue
        diff = query[q[b] : q[b + 1], None] - points[None, p[b] : p[b + 1]]
        squared = diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2
        best = squared.min(1, keepdim=True).values
        column = torch.arange(p[b + 1] - p[b]).expand_as(squared)
        index[q[b] : q[b + 1]] = p[b] + torch.where(squared == best, column, 1 << 30).min(1).values
        distance[q[b] : q[b + 1]] = best.squeeze(1).sqrt()
    return index, distance


CASES = [((5,), (7,)), ((13, 0, 9), (4, 6, 0)), ((0, 31), (17, 40)), ((40, 8, 3), (1, 33, 50))]


@pytest.mark.parametrize("integer", [True, False])
@pytest.mark.parametrize("counts", CASES)
def test_reference_matches_brute_force(counts, integer):
    inputs = list(_events(*counts, integer))
    index, distance = nearest_neighbor(*inputs)
    expected_index, expected_distance = _brute_force(*inputs)
    assert index.dtype == torch.long and distance.dtype == torch.float32
    assert torch.equal(index, expected_index)
    assert torch.equal(distance, expected_distance)


def test_fused_requires_cuda_tensors():
    with pytest.raises(RuntimeError, match="fused=True"):
        nearest_neighbor(*_events((3,), (3,), True), fused=True)


@pytest.mark.gpu
@pytest.mark.parametrize("integer", [True, False])
@pytest.mark.parametrize(
    "counts", CASES + [((1, 65, 129, 1000), (64, 1, 300, 2049)), ((4097, 0), (5000, 7))]
)
def test_fused_matches_reference(counts, integer):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    inputs = list(_events(*counts, integer, seed=1, device="cuda"))
    index, distance = nearest_neighbor(*inputs, fused=True)
    expected_index, expected_distance = nearest_neighbor(*inputs)
    torch.testing.assert_close(distance, expected_distance, rtol=1e-6, atol=0)
    if integer:
        assert torch.equal(index, expected_index)
    else:  # a fused multiply-add can round a near tie the other way: the chosen point must be as near
        query, _, points, _ = inputs
        found = index >= 0
        assert torch.equal(found, expected_index >= 0)
        chosen = (query[found] - points[index[found]]).norm(dim=1)
        torch.testing.assert_close(chosen, expected_distance[found], rtol=1e-6, atol=0)

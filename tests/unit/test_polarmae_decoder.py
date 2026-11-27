import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("decoder_use_kv", [True, False])
def test_both_decoders_run_forward_and_backward(decoder_use_kv):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from pimm.models.polarmae.polarmae import PoLArMAE

    torch.manual_seed(0)
    kw = dict(postnorm=True, add_pos_at_every_layer=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0)
    model = PoLArMAE(arch="vit_tiny", num_channels=4, voxel_size=5, masking_ratio=0.6, transformer_kwargs=kw,
                     decoder_arch="vit_tiny", decoder_kwargs=dict(transformer_kwargs=dict(depth=2, **kw)),
                     decoder_use_kv=decoder_use_kv).cuda()
    counts = torch.tensor([3000, 2000], device="cuda")
    coord = torch.rand(int(counts.sum()), 3, device="cuda") * 2 - 1
    feat = torch.cat([coord, torch.rand(len(coord), 1, device="cuda")], 1)
    loss = model({"feat": feat, "offset": counts.cumsum(0)})["loss"]
    loss.backward()
    assert torch.isfinite(loss)

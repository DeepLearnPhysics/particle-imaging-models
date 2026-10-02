# Models

pimm has two pretrained model families for LArTPC point clouds. Each release on Hugging Face is a portable export that `pimm.from_pretrained` loads in one call.

| Model | Hugging Face repository | Returns | Classes |
|---|---|---|---|
| Panda Base | `DeepLearnPhysics/Panda-Base` | per-point features (encoder only) | — |
| Panda Semantic | `DeepLearnPhysics/Panda-Semantic` | `seg_logits`, one row per point | shower, track, Michel, delta, low-energy deposit |
| Panda Particle | `DeepLearnPhysics/Panda-Particle` | particle masks and classes | photon, electron, muon, pion, proton |
| Panda Interaction | `DeepLearnPhysics/Panda-Interaction` | interaction masks | — |
| PoLAr-MAE Pretrain | `DeepLearnPhysics/PoLAr-MAE-Pretrain` | masked-reconstruction losses | — |
| PoLAr-MAE Semantic | `DeepLearnPhysics/PoLAr-MAE-Semantic` | `seg_logits`, one row per point | shower, track, Michel, delta |

```python
import pimm

model = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", device="cuda")
```

An export rebuilds the architecture and loads the weights. It doesn't include the input pipeline; [Panda](panda.md) and [PoLAr-MAE](polarmae.md) list the transforms each model's recipes use.

```{toctree}
panda
polarmae
run
fine-tune
export
```

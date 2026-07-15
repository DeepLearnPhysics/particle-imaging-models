# Models

Choose a published model by the output you need. Every repository below is a
portable pimm export: {py:func}`~pimm.from_pretrained` downloads the weights,
rebuilds the architecture from `config.json`, loads strictly, switches to eval
mode, and returns the model.

```python
import pimm

model = pimm.from_pretrained(
    "DeepLearnPhysics/Panda-Semantic",
    device="cuda",
)
```

:::{important}
The portable loader reconstructs the **architecture**, not an executable data
pipeline. An export made from a full run may retain a sanitized `data` section,
but `from_pretrained` does not build or apply it. Reproduce the coordinate,
energy, sampling, and collation steps from the release recipe before calling a
model. See [Prepare an input batch](pretrained.md#prepare-an-input-batch).
:::

## Choose a model

| Goal | Model | What `forward` returns in eval mode | Published metric |
|---|---|---|---|
| Build a Panda task model | [`DeepLearnPhysics/Panda-Base`](https://huggingface.co/DeepLearnPhysics/Panda-Base) | A {py:class}`Point <pimm.models.utils.structure.Point>` representation from an encoder-only {py:class}`PT-v3m2 <pimm.models.point_transformer_v3.point_transformer_v3m2_sonata.PointTransformerV3>`; no task prediction | Not applicable |
| Label every point | [`DeepLearnPhysics/Panda-Semantic`](https://huggingface.co/DeepLearnPhysics/Panda-Semantic) | `seg_logits`, shape $(N, 5)$ | See TODO below |
| Find particles and assign PID | [`DeepLearnPhysics/Panda-Particle`](https://huggingface.co/DeepLearnPhysics/Panda-Particle) | Query masks/classes; {py:meth}`~pimm.models.panda_detector.detector_v4.UnifiedDetector.postprocess` produces point-level instances and PID | See TODO below |
| Group points by interaction | [`DeepLearnPhysics/Panda-Interaction`](https://huggingface.co/DeepLearnPhysics/Panda-Interaction) | Query masks/classes; {py:meth}`~pimm.models.panda_detector.detector_v4.UnifiedDetector.postprocess` produces point-level interaction instances | See TODO below |
| Continue PoLAr-MAE pretraining | [`DeepLearnPhysics/PoLAr-MAE-Pretrain`](https://huggingface.co/DeepLearnPhysics/PoLAr-MAE-Pretrain) | Self-supervised loss terms, not a downstream prediction | Not applicable |
| Run four-class PoLAr-MAE segmentation | [`DeepLearnPhysics/PoLAr-MAE-Semantic`](https://huggingface.co/DeepLearnPhysics/PoLAr-MAE-Semantic) | `seg_logits`, shape $(N, 4)$ | $\mathrm{mF1} \approx 0.82$, as reported by the model card |

$N$ is the total number of points in the packed batch. The Hub repository names
above use their canonical capitalization; Hub lookup itself is case-insensitive.

The two maintained collections are the quickest way to see new releases:

- [Panda collection](https://huggingface.co/collections/DeepLearnPhysics/panda)
- [PoLAr-MAE collection](https://huggingface.co/collections/DeepLearnPhysics/polar-mae)

:::{admonition} TODO
:class: pimm-todo
Add signed-off, held-out Panda metrics and their evaluation protocols to the
model cards, then replace the three TODO cells above. Each value needs its split,
preprocessing contract, and metric definition.
:::

## Output and label lookup

| Model | Registry type | Labels or objective | Next step |
|---|---|---|---|
| Panda Base | `PT-v3m2` | Masked-point pretrained encoder; no task head | Attach a head and fine-tune |
| Panda Semantic | {py:class}`DefaultSegmentorV2 <pimm.models.default.DefaultSegmentorV2>` | shower, track, Michel, delta, LED | `output["seg_logits"].argmax(-1)` |
| Panda Particle | {py:class}`detector-v4 <pimm.models.panda_detector.detector_v4.UnifiedDetector>` | photon, electron, muon, pion, proton, LED | {py:meth}`~pimm.models.panda_detector.detector_v4.UnifiedDetector.postprocess` |
| Panda Interaction | {py:class}`detector-v4 <pimm.models.panda_detector.detector_v4.UnifiedDetector>` | stuff, thing | {py:meth}`~pimm.models.panda_detector.detector_v4.UnifiedDetector.postprocess` |
| PoLAr-MAE Pretrain | {py:class}`PoLAr-MAE <pimm.models.polarmae.polarmae.PoLArMAE>` | masked geometry reconstruction and energy infilling | Read `loss`, `chamfer_loss`, `energy_loss` |
| PoLAr-MAE Semantic | {py:class}`PoLArMAE-SemSeg <pimm.models.polarmae.polarmae_semseg.PoLArMAESemSeg>` | shower, track, Michel, delta | `output["seg_logits"].argmax(-1)` |

The model cards are the authority for provenance and intended use. The recipe
configs are the authority for preprocessing and training settings.

### Open the matching recipe

| Model | In-repository recipe |
|---|---|
| Panda Base | [`pretrain-sonata-v1m1-pilarnet-smallmask.py`](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/configs/panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py) |
| Panda Semantic | [`semseg-pt-v3m2-pilarnet-ft-5cls-fft.py`](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py) |
| Panda Particle | [`detector-v4-pt-v3m2-ft-pid-fft.py`](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/configs/panda/panseg/detector-v4-pt-v3m2-ft-pid-fft.py) |
| Panda Interaction | [`detector-v4-pt-v3m2-ft-vtx-fft.py`](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/configs/panda/panseg/detector-v4-pt-v3m2-ft-vtx-fft.py) |
| PoLAr-MAE Pretrain | [`pretrain-polarmae-pilarnet.py`](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/configs/polarmae/pretrain-polarmae-pilarnet.py) |
| PoLAr-MAE Semantic | [`semseg-polarmae-pilarnet-fft-reproduce.py`](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/configs/polarmae/semseg/semseg-polarmae-pilarnet-fft-reproduce.py) |

## Next

| Goal | Guide |
|---|---|
| Run inference or fine-tune released weights | {doc}`Load, run, and fine-tune a model <pretrained>` |
| Publish portable weights | {doc}`Export and publish <export>` |

```{toctree}
:hidden:

pretrained
export
```

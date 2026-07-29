# ADR 0002: backend-neutral public boundary

- Status: accepted
- Date: 2026-07-28
- Milestone: 00

## Context

WarpConvNet provides sparse geometry and spatial operators needed by retained
PIMM models. Making its geometry objects PIMM's universal dataset output would
couple every modality, collator, model, and task to one backend.

PIMM must also continue to support dense, graph, image, token, and
project-local research models.

## Decision

WarpConvNet is the primary spatial backend, not PIMM's public data ontology.
Datasets emit backend-neutral values. Conversion to WarpConvNet geometry occurs
at an explicit backend/model seam.

An ordinary `torch.nn.Module` remains a valid trainer-facing model. It must
accept the configured batch and return a mapping containing scalar
`output["loss"]` during training. No PIMM or WarpConvNet base class is required.

PTv3 m2 and m8 remain first-class profiles. LitePT uses v1m2 semantics. Sparse
U-Net remains dimension-generic over supported 2D and 3D convolution. Volt and
Panda are consolidated without retaining copied implementations as presets.

## Consequences

- Typed public batches are introduced before the WCN conversion seam.
- Multiple sparse 2D planes remain distinct modalities.
- Custom models may ignore WCN.
- PIMM wrappers cover stable conversion boundaries, not the complete WCN API.
- No WarpConvNet dependency is introduced in Milestone 00.

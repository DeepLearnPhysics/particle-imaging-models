# Migration contract fixtures

These fixtures contain deterministic synthetic values only. They freeze
observable pre-WarpConvNet behavior without depending on detector data,
checkpoints, network access, CUDA kernels, or private repository content.

- `legacy_packed_batch.yaml` freezes the current recursive collator's packed
  tensor, cumulative offset, string metadata, and private-key behavior.
- `plain_module_step.yaml` freezes the minimum ordinary `torch.nn.Module`
  forward/loss contract used at the trainer boundary.

If an intentional migration changes a fixture-facing representation, keep the
legacy fixture for compatibility testing and add the new contract separately.

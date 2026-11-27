"""The one error a fused operation raises when it can't run a call."""


def require_fused(op, reason):
    """Raise if ``reason``, why ``op``'s fused version can't run this call, is set."""
    if reason is not None:
        raise RuntimeError(
            f"pimm.ops.{op}(fused=True) can't run: {reason}. Use fused=False for the original code."
        )

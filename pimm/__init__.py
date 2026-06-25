"""pimm — particle imaging models!

Author: Samuel Young
Email: youngsam@stanford.edu

Please cite this codebase if you find it useful.
"""

# Lazily expose the high-level model IO helpers as `pimm.from_pretrained`, etc.,
# without importing the (heavier) export stack at `import pimm` time.
_EXPORT_API = {"from_pretrained", "save_pretrained", "push_to_hub"}


def __getattr__(name):
    if name in _EXPORT_API:
        from pimm import export

        return getattr(export, name)
    raise AttributeError(f"module 'pimm' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_EXPORT_API))

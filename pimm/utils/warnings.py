"""Warnings emitted by pimm's public APIs."""

from __future__ import annotations

import threading
import warnings as _warnings
from collections.abc import Callable, Hashable
from typing import TypeVar

from typing_extensions import deprecated as _typing_deprecated


class PimmWarning(UserWarning):
    """Base category for warnings emitted by pimm."""


class PimmDeprecationWarning(FutureWarning):
    """Warning for pimm APIs that will be removed in a future release."""


_seen: set[Hashable] = set()
_seen_lock = threading.Lock()
_CallableT = TypeVar("_CallableT", bound=Callable)


def warn(
    message: str,
    category: type[Warning] = PimmWarning,
    *,
    stacklevel: int = 2,
) -> None:
    """Emit a pimm warning attributed to the caller."""
    _warnings.warn(message, category, stacklevel=stacklevel)


def warn_once(
    message: str,
    category: type[Warning] = PimmWarning,
    *,
    key: Hashable | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a warning once per process for the given key."""
    warning_key = key if key is not None else (category, message)
    with _seen_lock:
        if warning_key in _seen:
            return
        _seen.add(warning_key)
    warn(message, category, stacklevel=stacklevel + 1)


def deprecation_message(feature: str, replacement: str, remove_in: str) -> str:
    """Build the standard pimm deprecation message."""
    return (
        f"{feature} is deprecated; use {replacement} instead. "
        f"It will be removed in pimm {remove_in}."
    )


def warn_deprecated(
    feature: str,
    replacement: str,
    remove_in: str,
    *,
    stacklevel: int = 2,
) -> None:
    """Warn that a pimm feature is scheduled for removal."""
    warn(
        deprecation_message(feature, replacement, remove_in),
        PimmDeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def deprecated(*, replacement: str, remove_in: str) -> Callable[[_CallableT], _CallableT]:
    """Mark a callable as deprecated and warn whenever it is called."""

    def decorate(obj: _CallableT) -> _CallableT:
        message = deprecation_message(obj.__qualname__, replacement, remove_in)
        return _typing_deprecated(
            message,
            category=PimmDeprecationWarning,
            stacklevel=2,
        )(obj)

    return decorate

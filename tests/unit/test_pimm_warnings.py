import warnings

from pimm.utils import (
    PimmDeprecationWarning,
    PimmWarning,
    deprecated,
    warn,
    warn_deprecated,
    warn_once,
)


def test_warning_categories_and_warn_once():
    key = object()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn("ordinary")
        warn_once("only once", key=key)
        warn_once("only once", key=key)

    assert [item.category for item in caught] == [PimmWarning, PimmWarning]
    assert [str(item.message) for item in caught] == ["ordinary", "only once"]


def test_deprecation_helpers_use_visible_pimm_category():
    @deprecated(replacement="new_api", remove_in="0.6.0")
    def old_api(value):
        return value + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated("old setting", "new setting", "0.6.0")
        assert old_api(2) == 3

    assert len(caught) == 2
    assert all(item.category is PimmDeprecationWarning for item in caught)
    assert all("pimm 0.6.0" in str(item.message) for item in caught)
    assert old_api.__deprecated__

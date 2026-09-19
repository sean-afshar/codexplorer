"""The pre-merge packages still import, with a deprecation warning pointing at connexplorer."""


def test_legacy_packages_warn():
    import importlib
    import warnings

    for pkg in ("shayan", "cex"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.reload(importlib.import_module(pkg))
        assert any(issubclass(x.category, DeprecationWarning) and "connexplorer" in str(x.message) for x in w), pkg

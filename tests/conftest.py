import pytest


def pytest_collection_modifyitems(config, items):
    markexpr = getattr(config.option, "markexpr", "") or ""
    if "smoke" in markexpr:
        return

    skip_smoke = pytest.mark.skip(
        reason="smoke tests are skipped by default; use pytest -m smoke to run them"
    )
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip_smoke)

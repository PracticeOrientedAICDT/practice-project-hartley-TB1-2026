"""Enforce 0.1s unit budget; auto-mark every test with `@pytest.mark.unit`."""

import time

import pytest

UNIT_BUDGET_S = 0.1


def pytest_collection_modifyitems(items):
    """Tag unit tests by path."""
    for item in items:
        if item.nodeid.startswith("tests/unit/"):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def _enforce_unit_budget():
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed < UNIT_BUDGET_S, f"Unit test too slow: {elapsed:.3f}s >= {UNIT_BUDGET_S:.3f}s budget"

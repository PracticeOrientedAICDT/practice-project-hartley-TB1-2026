"""Fail any unit test slower than the budget.

Single autouse fixture: applies to everything under tests/unit/
with no per-test decoration needed.

Unit mark is applied automatically via collection hook below —
do not add `@pytest.mark.unit` manually.
"""

import time

import pytest

UNIT_BUDGET_S = 0.1


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/unit/"):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def _enforce_unit_budget():
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed < UNIT_BUDGET_S, (
        f"Unit test too slow: {elapsed:.3f}s >= {UNIT_BUDGET_S:.3f}s budget"
    )

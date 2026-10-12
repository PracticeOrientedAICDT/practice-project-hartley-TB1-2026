"""Auto-mark every test under tests/integration/.

No per-test `@pytest.mark.integration` needed.
"""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/integration/"):
            item.add_marker(pytest.mark.integration)

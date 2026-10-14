"""Guard that every collected test has a type mark.

Folder conftests auto-apply `unit` / `integration`, so this only fails
for tests collected outside those folders (e.g. a new tests/e2e/
without its own hook).
"""

import pytest


@pytest.fixture(autouse=True)
def _require_type_mark(request):
    names = {m.name for m in request.node.iter_markers()}
    assert names & {"unit", "integration"}, (
        f"{request.node.nodeid} has no type mark — put it under tests/unit,"
        " tests/integration or unit/scripts"
    )

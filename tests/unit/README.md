# Unit tests

Pure, deterministic, fast. Budgets: **<0.1s per test**, enforced by an
autouse fixture in `conftest.py` (no per-test wrapper needed).

Rules:
- No network, filesystem, DB, subprocess, or `sleep`.
- Mock at module boundaries; one behaviour per test.
- Mark every test with `@pytest.mark.unit`.

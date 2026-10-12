# Integration tests

Cross-module / wiring tests. Real IO allowed. Slower than unit —
no per-test time budget.

Rules:
- No manual marking needed — `conftest.py` auto-applies `@pytest.mark.integration`.
- Prefer hermetic fixtures; document external dependencies in the test.

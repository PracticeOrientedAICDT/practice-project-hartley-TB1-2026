# Integration tests

Cross-module / wiring tests. Real IO allowed. Slower than unit —
no per-test time budget.

Rules:
- Every test automatically marked with `@pytest.mark.integration`.
- Prefer hermetic fixtures; document external dependencies in the test.

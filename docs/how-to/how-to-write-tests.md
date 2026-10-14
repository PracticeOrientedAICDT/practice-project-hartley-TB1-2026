# Strict Unit Test definition:

A unit test is a fast, deterministic, independently executable automated test that verifies one specific behaviour of a single unit of Python code, using controlled inputs and asserting an observable outcome. All dependencies outside the unit under test must be replaced with test doubles, and the test must not rely on external services, shared mutable state, or execution order.

https://martinfowler.com/bliki/UnitTest.html

# Strict Integration Test definition: 

An integration test is an automated test that verifies the correct interaction between two or more software components across a defined interface, using the real implementation of the integration boundary and any dependencies necessary to validate that interaction. The test must assert an observable outcome and be repeatable and independent of execution order.

https://martinfowler.com/bliki/IntegrationTest.html

# General Principles

- pytest is used for our testing suite, avoid using unittest
- execute all tests via `uv run pytest` or inidividual folders via `uv run pytest -m tests/unit`
- name tests `test_that_{unit}_{verb}_{behaviour}` (e.g. `test_that_check_pr_template_rejects_invalid_body`)

https://docs.pytest.org/en/stable/index.html

https://martinfowler.com/articles/practical-test-pyramid.html

# Unit test principles

- Common use cases test individual methods *def XXX()*
- Often use pytest.mark.parametrize to iterate through multiple cases on each unit test
- TBD ....

# Integration test principles

- Common use cases test pairs of python modules 
- TBD ....
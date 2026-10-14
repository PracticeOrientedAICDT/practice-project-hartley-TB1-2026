"""Tests for the PR template check used by the pass/fail CI gate."""

import pytest

from scripts.check_pr_template import check_pr_template

PASSING = [
    """
## What / why

Refactors the gate so the pytest-unit marker word shows up in prose and the
logic stays covered by unit tests.

Closes #12

## Target branch check

- [x] Base branch is correct (`feature/*` → `dev`)

## Test evidence

- [x] `lint` passes (`ruff check`, `ruff format --check`, `ty check`)
- [x] `pytest-unit` passes
- [x] `pytest-integration` passes

## Reviewer checklist (for the reviewer)

- [ ] CI is green on the latest commit
""",
    """
## Test evidence

- [X] `pytest-unit` passes

## What / why

A sufficiently long explanation of the change and the reason it is needed.

closes #7
""",
    """
## What / why

<!-- Keep this section focused on the reason for the change. -->

A sufficiently long explanation of the change and the reason it is needed.

Closes #12

## Test evidence

- [x] `lint` passes
- [x] `pytest-unit` passes
- [x] `pytest-integration` passes (N/A — docs only, no integration surface)
""",
]

FAILING = [
    (
        """
## What / why

A sufficiently long explanation of the change and the reason it is needed.

Closes #

## Test evidence

- [x] `pytest-unit` passes
""",
        ['Missing "Closes #N" with a real issue number.'],
    ),
    (
        """
## What / why

## Test evidence

- [x] `pytest-unit` passes

Closes #12
""",
        ['"## What / why" needs prose (30+ characters beyond the Closes line).'],
    ),
    (
        "Closes #12",
        [
            'Missing "## What / why" section.',
            'No tickable boxes under "## Test evidence".',
        ],
    ),
    (
        """
## What / why

A sufficiently long explanation of the change and the reason it is needed.

Closes #12

## Test evidence

- [x] `lint` passes
- [ ] `pytest-unit` passes
""",
        ["Unticked test-evidence box."],
    ),
    (
        "",
        [
            'Missing "Closes #N" with a real issue number.',
            'Missing "## What / why" section.',
            'No tickable boxes under "## Test evidence".',
        ],
    ),
]


@pytest.mark.parametrize("body", PASSING)
def test_pr_template_passes(body):
    """A template-shaped body passes with no failure messages."""
    status, messages = check_pr_template(body)
    assert status is True
    assert messages == []


@pytest.mark.parametrize("body, expected", FAILING)
def test_pr_template_fails(body, expected):
    """Each violation fails with exactly the gate's failure message(s)."""
    status, messages = check_pr_template(body)
    assert status is False
    assert messages == expected

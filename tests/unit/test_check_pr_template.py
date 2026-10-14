"""Tests for the PR template check used by the pass/fail CI gate."""

from scripts.check_pr_template import check

GOOD_BODY = """
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
"""


def test_good_body_passes():
    assert check(GOOD_BODY) == []


def test_bare_closes_placeholder_fails():
    body = """
## What / why

A sufficiently long explanation of the change and the reason it is needed.

Closes #

## Test evidence

- [x] `pytest-unit` passes
"""
    assert check(body) == ['Missing "Closes #N" with a real issue number.']


def test_missing_what_why_section_fails():
    body = """
Closes #12

## Test evidence

- [x] `pytest-unit` passes
"""
    assert check(body) == ['Missing "## What / why" section.']


def test_what_why_with_only_the_closes_line_fails():
    body = """
## What / why

Closes #12

## Test evidence

- [x] `pytest-unit` passes
"""
    assert check(body) == ['"## What / why" needs prose (30+ characters beyond the Closes line).']


def test_what_why_with_short_prose_fails():
    body = """
## What / why

Fixes it.

Closes #12

## Test evidence

- [x] `pytest-unit` passes
"""
    assert check(body) == ['"## What / why" needs prose (30+ characters beyond the Closes line).']


def test_unticked_test_evidence_box_fails():
    body = """
## What / why

A sufficiently long explanation of the change and the reason it is needed.

Closes #12

## Test evidence

- [x] `lint` passes
- [ ] `pytest-unit` passes
"""
    assert check(body) == ["Unticked test-evidence box."]


def test_missing_test_evidence_section_fails():
    body = """
## What / why

A sufficiently long explanation of the change and the reason it is needed.

Closes #12

## Reviewer checklist (for the reviewer)

- [ ] CI is green
"""
    assert check(body) == ['No tickable boxes under "## Test evidence".']


def test_na_rewrite_of_a_box_line_ticked_passes():
    body = """
## What / why

A sufficiently long explanation of the change and the reason it is needed.

Closes #12

## Test evidence

- [x] `lint` passes
- [x] `pytest-unit` passes
- [x] `pytest-integration` passes (N/A — docs only, no integration surface)
"""
    assert check(body) == []

"""Fail CI when a PR body does not adhere to .github/pull_request_template.md.

Reads the PR body from stdin, checks:
  1. A ``Closes #N`` link with a real issue number.
  2. ``## What / why`` and ``## Test evidence`` sections exist and are non-empty.
  3. The three test-evidence checkboxes are ticked
     (lint, pytest-unit, pytest-integration).

The reviewer checklist is intentionally ignored -- that is for the
reviewer, not the author. Prints each failure; exits 0 on pass, 1 on fail.
"""

import re
import sys


def section_text(body: str, heading: str) -> str | None:
    """Return comment-stripped text under ``## <heading>``, or None if missing."""
    match = re.search(
        r"^##\s+" + heading + r"\s*$(.*?)(?=^##\s+|\Z)",
        body,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        return None
    return re.sub(r"<!--.*?-->", "", match.group(1), flags=re.DOTALL).strip()


def main() -> int:
    body = sys.stdin.read()
    failures = []

    # 1. Linked issue with a real number (bare "Closes #" is the template placeholder).
    if not re.search(r"closes\s+#\d+", body, re.IGNORECASE):
        failures.append('Missing "Closes #N" with a real issue number.')

    # 2. Required sections present and non-empty.
    for heading in (r"What\s*/\s*why", r"Test evidence"):
        content = section_text(body, heading)
        if content is None:
            failures.append(f'Missing "## {heading}" section.')
        elif not content:
            failures.append(f'"## {heading}" section is empty.')

    # 3. Test-evidence boxes ticked. An N/A still needs a tick plus a reason
    #    written next to it -- the reviewer judges whether the reason holds.
    for label, marker in (
        ("lint", "ruff check"),
        ("pytest-unit", "pytest-unit"),
        ("pytest-integration", "pytest-integration"),
    ):
        lines = [line for line in body.splitlines() if marker in line]
        if not lines:
            failures.append(f'Test-evidence line for "{label}" is missing.')
        elif not all(re.match(r"\s*-\s*\[[xX]\]", line) for line in lines):
            failures.append(f'Test-evidence box for "{label}" is unticked.')

    if failures:
        print("PR template check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PR template check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Check a pull request body against the repo's PR template.

The pure logic lives in :func:`check`; :func:`main` reads the PR body from
stdin so the same code path runs in CI and in the unit tests.
"""

from __future__ import annotations

import re
import sys

PROSE_MIN_CHARS = 30

_COMBINED_FLAGS = re.IGNORECASE | re.MULTILINE | re.DOTALL
_CLOSES = re.compile(r"closes\s+#\d+", re.IGNORECASE)
_WHY_SECTION = re.compile(r"^##\s+What\s*/\s*why\s*$(.*?)(?=^##\s+|\Z)", _COMBINED_FLAGS)
_TEST_SECTION = re.compile(r"^##\s+Test evidence\s*$(.*?)(?=^##\s+|\Z)", _COMBINED_FLAGS)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CHECKBOX_LINE = re.compile(r"^\s*-\s*\[[ xX]\]")
_CHECKBOX = re.compile(r"^\s*-\s*\[([ xX])\]", re.MULTILINE)


def check(body: str) -> list[str]:
    """Return failure messages for a PR body; an empty list means it passes."""
    failures: list[str] = []

    if not _CLOSES.search(body):
        failures.append('Missing "Closes #N" with a real issue number.')

    why = _WHY_SECTION.search(body)
    if why is None:
        failures.append('Missing "## What / why" section.')
    else:
        prose = _HTML_COMMENT.sub("", why.group(1))
        prose = _CLOSES.sub("", prose)
        prose = "\n".join(line for line in prose.splitlines() if not _CHECKBOX_LINE.match(line))
        if len(prose.strip()) < PROSE_MIN_CHARS:
            failures.append('"## What / why" needs prose (30+ characters beyond the Closes line).')

    section = _TEST_SECTION.search(body)
    boxes = _CHECKBOX.findall(section.group(1)) if section else []
    if not boxes:
        failures.append('No tickable boxes under "## Test evidence".')
    elif any(box == " " for box in boxes):
        failures.append("Unticked test-evidence box.")

    return failures


def main() -> int:
    """Read a PR body from stdin, print the verdict, and return an exit code."""
    failures = check(sys.stdin.read())
    if failures:
        print("PR template check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PR template check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# How to make a PR

Branch protection can't be enabled on this repo (private, GitHub Free), so
reviewers enforce these rules by convention.

## Branches

- `main` — release-ready. Never commit to it directly.
- `dev` — integration. Never commit to it directly.
- `feature/<name>` — your work. Branch off `dev`, one purpose per branch.

## Rules

- All changes go through a PR. No direct pushes to `main` or `dev`.
- Feature PRs target `dev`. Only `dev` targets `main` (releases).
- Merge needs: CI green (`lint`, `pytest-unit`, `pytest-integration`),
  one approval from someone other than the author, and no unresolved threads.
- Re-request review after material changes.

## Steps

```sh
git checkout dev && git pull
git checkout -b feature/<name>
# ... commit ...
git push -u origin feature/<name>
```

Open a PR (body template fills in automatically), request a reviewer,
address feedback, then squash-merge and delete the branch.

## Releases (`dev` → `main`)

Open when `dev` is green. Same bar: full CI + one non-author approval.
Title it `Release: <date or scope>`.

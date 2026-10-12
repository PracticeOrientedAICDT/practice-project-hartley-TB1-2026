# How to make a PR

Branches `main` and `dev` are protected by repository rulesets
(`protect-main`, `protect-dev`). The rules below are enforced by GitHub —
not just by convention.

## Branches

- `main` — release-ready. Never commit to it directly. Never delete it.
- `dev` — integration. Never commit to it directly. Never delete it.
- `feature/<name>` — your work. Branch off `dev`, one purpose per branch.
  Feature branches auto-delete after merge; `main`/`dev` are blocked from
  deletion and force-pushes by the rulesets.

## Rules (enforced)

- All changes go through a PR. No direct pushes to `main` or `dev`
  (blocked by ruleset; admins bypass only for emergencies).
- Feature PRs target `dev`. Only `dev` targets `main` (releases).
- Merge needs: CI green (`lint`, `pytest-unit`, `pytest-integration`)
  on the latest commit with branch up-to-date, one approval from someone
  other than the author (self-approval never counts), and no unresolved
  threads. New pushes dismiss stale approvals and need re-review.
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

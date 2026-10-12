# How to make a PR

This repo uses three branch tiers. Server-side branch protection is currently
**unavailable** (private repo on GitHub Free — the protection API returns
`403`), so these rules are enforced by **reviewers and convention**, not by
GitHub settings. Follow them as if they were hard rules.

## Branch roles

| Branch | Purpose | Push directly? |
|---|---|---|
| `main` | Release-ready code only. | Never. |
| `dev` | Integration branch. Feature work lands here first. | Never. |
| `feature/<short-name>` | Day-to-day work. One purpose per branch. | Yes (your own branch). |

## Golden rules

1. **Never commit directly to `main` or `dev`.** All changes arrive via a PR.
2. **Feature branches branch off `dev`** and merge back into `dev`:
   `feature/*` → `dev`.
3. **Only `dev` merges into `main`** (release PRs). A PR from a
   `feature/*` branch targeting `main` must be closed and retargeted at `dev`.
   Reviewers reject any `feature/*` → `main` PR.
4. **No merge without all of the following:**
   - All CI checks green: `lint`, `pytest-unit`, `pytest-integration`
     (see `.github/workflows/ci.yml`).
   - **One approving review from someone other than the author.**
     You cannot approve your own PR.
   - Branch up to date with the base; no unresolved conversations.
5. After material changes, **re-request review** — stale approvals don't count.

## Step by step

1. **Start from `dev`:**
   ```sh
   git checkout dev
   git pull
   git checkout -b feature/<short-name>
   ```
2. **Make small, focused commits** on your feature branch.
3. **Push your branch:**
   ```sh
   git push -u origin feature/<short-name>
   ```
4. **Open a PR** against `dev` (or against `main` only if it is a
   `dev` → `main` release). The PR body is pre-filled from
   `.github/pull_request_template.md` — fill in every section:
   what/why, test evidence, risk.
5. **Request a reviewer** (a teammate, never yourself). CI must go green
   and they must approve before merge.
6. **Address feedback**, push new commits, re-request review.
7. **Merge** (squash-merge preferred), then **delete the feature branch**.

## Release PRs (`dev` → `main`)

- Open only when `dev` is green and reviewed.
- Title clearly, e.g. `Release: <date or scope>`.
- Same bar as above: full CI green + one non-author approval.
- After merging into `main`, `dev` stays as-is (it is already contained).

## Future: server-side protection

If the org upgrades to GitHub Team (or the repo is made public), apply this
spec via repository rulesets / branch protection on `main` and `dev`:

- Require pull request before merging; block direct pushes, force-pushes,
  and branch deletion.
- Require 1 approving review; dismiss stale reviews on new commits;
  require review from someone other than the author.
- Require status checks to pass: `lint`, `pytest-unit`,
  `pytest-integration`.
- Additionally document that only `dev` may target `main` (enforced by
  reviewers, since GitHub cannot restrict PR source branches natively).

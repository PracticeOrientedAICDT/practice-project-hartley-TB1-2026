# Branch rules

Enforced by repository rulesets `protect-main` (`refs/heads/main`) and
`protect-dev` (`refs/heads/dev`). Enforcement: `active`.
`feature/*` branches are not protected.

## Rules (both rulesets)

- `deletion`: blocked. `main`/`dev` can never be deleted.
- `non_fast_forward`: blocked. No force-pushes to `main`/`dev`.
- `pull_request`:
  - `required_approving_review_count: 1` (self-approval never counts;
    PR author cannot approve own PR).
  - `dismiss_stale_reviews_on_push: true` — new commits dismiss approvals.
  - `require_last_push_approval: true` — last push must be approved by
    someone other than the pusher.
  - `require_code_owner_review: false` — no CODEOWNERS in this repo.
  - `required_review_thread_resolution: true` — no unresolved threads.
  - `allowed_merge_methods: [merge, squash, rebase]` — all three allowed.
- `required_status_checks` (strict):
  - `lint`, `pytest-unit`, `pytest-integration` must pass.
  - `strict_required_status_checks_policy: true` — branch must be
    up-to-date with base before merge.

## Bypass (emergencies only)

- `OrganizationAdmin: always`
- `RepositoryRole admin (id 5): always`
- Everyone else: never. Direct pushes to `main`/`dev` are rejected;
  use a PR. See `docs/how-to/how-to-make-a-pr.md`.

## Supporting settings

- CI (`.github/workflows/ci.yml`): `push` on `main`, `dev`, `feature/**`
  + `pull_request` on `main`, `dev`, with `cancel-in-progress: true`.
- Repo: `delete_branch_on_merge: true` — `feature/*` heads auto-delete.
  `main`/`dev` cannot auto-delete (deletion rule above).
- PR template: `.github/pull_request_template.md` (auto-fills, enforced
  by convention + reviewer checklist, no bot check).

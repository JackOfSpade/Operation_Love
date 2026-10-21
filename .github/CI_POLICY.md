# CI and merging

Changes use a temporary branch and a pull request into `main`. The public
repository's branch protection requires the stable checks listed below and an
up-to-date branch. A failed or skipped prerequisite fails its aggregate check.

- `CI required`
- `Security required`
- `Dependency review required`

The **Merge validated PR** workflow is manual and accepts one PR number. It
checks the current main ancestry, exact PR head, GitHub Actions check origin,
and latest required results before merging. It checks out trusted main policy,
never PR code. It dispatches main validation after the merge and never deletes
branches itself. GitHub's existing repository auto-delete setting can separately
delete the selected PR's merged head branch. Existing branches and PRs are not
swept or automatically adopted.

On GitHub Free, private repository branch protection is unavailable. In that
case this workflow's explicit checks guard this merge path; other writers are
not protected by server-side rules. Standard public runners are used, and
existing cloud/API workflows retain their own credentials and cost controls.

The full Linux suite runs on PRs targeting main, pushes to main, and manual
dispatch. Six independent standard Linux runners partition the full collected
pytest node IDs deterministically, including cases within the slow capture
module. Each runner executes its selected cases from one module in a fresh
serial process; only a SIGKILL receives the existing per-case process fallback.

Each shard uploads its full inventory digest, selected IDs, and actual pytest
phase outcomes with one-day retention. A required reconciliation job verifies
the same source commit and inventory, all six distinct shards, and a disjoint
union covering every collected case with complete successful pytest outcomes.
Missing artifacts, collection errors, assertion failures, incomplete results,
and cancelled shards fail `CI required`. A failed-job rerun replaces only its
own shard artifact; the remaining same-run results must still match exactly.

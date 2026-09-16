# CI Workflows

`schedule:`, `issue_comment:`, and `workflow_run:` workflows must exist on
the repository's **default branch**. Check the current branch with
`gh repo view isaac-sim/IsaacLab --json defaultBranchRef`; do not assume it
is `main` or `develop`. `workflow_dispatch:` also requires the workflow to
exist on the default branch, but can then run a selected branch or tag.
`pull_request:` and `push:` workflows work on `develop` normally.

## Recovering pre-step runner disconnects

`recover-runner-disconnect.yml` runs on a GitHub-hosted runner after
`Docker + Tests` completes. It requests **at most one automatic retry**,
only after attempt 1 of a pull-request run, when every failed job:

- Uses the `self-hosted` and `gpu` labels.
- Has no recorded steps and has GitHub's job-level `.github` failure
  annotation beginning `The self-hosted runner lost communication with the server.`

Ordinary test failures, cancellations, timeouts, missing evidence, and
failures after steps started require manual investigation. The recovery
workflow checks every job, including failures tolerated by
`continue-on-error`, before requesting a failed-jobs rerun. GitHub reruns
the failed jobs **and their dependent jobs**; successful independent jobs
are retained. Attempt 2 is never retried automatically, including when
attempt 2 was started manually. The original failure stays in run history,
and a recovery summary records the run and job IDs.

The PR must still be open at the same head SHA and repository, and a newer
run on the same source branch suppresses recovery. The recovery workflow's
per-run concurrency group serializes duplicate events. A final state read
avoids retrying a run already restarted by a person; GitHub provides no
atomic compare-and-rerun API, so a push or manual rerun can still race the
final request. A failed POST is reported and is never blindly retried.

This is a `workflow_run` workflow with `actions: write`: it checks out only
its own trusted default-branch commit, never the triggering PR, and reads
no artifacts or caches. It uses `GITHUB_TOKEN`, not a new bot secret.

**Activation:** merge into `develop`, then install both the workflow and
`.github/scripts/recover-runner-disconnect.cjs` on the current default
branch through the normal review process. A merge into `develop` alone
does not activate recovery when another branch is default. Removing the
recovery workflow disables automatic retries; existing CI concurrency and
manual reruns are unchanged.

Run the dependency-free policy and orchestration tests locally with:

```sh
node --test .github/scripts/recover-runner-disconnect.test.cjs
```

The `Tools Tests` workflow runs these tests for changes to recovery files.
Step-level retry actions cannot repair this failure because no workflow
step starts on the disconnected runner.

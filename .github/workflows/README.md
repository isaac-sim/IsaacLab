# CI Workflows

`schedule:` and `workflow_dispatch:` triggers fire **only from the
repository's current default branch**, which is not always `main` — it is
`release/3.0.0-beta2` at the time of writing. A workflow YAML must live on
that branch for its cron to register; the same file on another branch has no
effect. `pull_request:` and `push:` triggers fire from the event branch's
file and work normally on `develop`; `pull_request_target:` fires from the
base branch's file, and unlike `pull_request` it carries a write token and
secrets on fork PRs, which is how a `develop`-resident workflow can act on a
PR. It must never check out PR code.

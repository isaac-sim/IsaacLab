# CI Workflows

`schedule:` triggers run only from the repository's current default branch.
A scheduled workflow YAML must live on that branch for its cron to register;
the same schedule on another branch has no effect. A `workflow_dispatch:`
workflow must also exist on the default branch, although a manual dispatch can
select another branch or tag. `pull_request:` and `push:` triggers use the
event branch normally.

## Changing the Default Branch

Before changing the repository default branch:

1. Confirm every scheduled or manually dispatched workflow exists on the new
   default branch and explicitly checks out any non-default target it operates
   on.
2. Protect the new default branch with the required reviews and status checks.
   If the nightly changelog workflow targets it, add `isaaclab-bot` to the
   ruleset bypass list so its generated commit can be pushed.
3. Allow the new default branch to deploy to the `github-pages` environment.
4. Confirm `docs.yaml` includes the new branch in `SMV_BRANCH_WHITELIST`. The
   docs build uses the repository default branch as `DOCS_DEFAULT_REF` and
   fails if that version was not built.
5. Manually dispatch the docs workflow after the settings change and verify
   the site root redirects to the new default version.

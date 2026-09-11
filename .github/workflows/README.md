# CI Workflows

`schedule:` triggers run only from the repository's current default branch.
A scheduled workflow YAML must live on that branch for its cron to register;
the same schedule on another branch has no effect. A `workflow_dispatch:`
workflow must also exist on the default branch, although a manual dispatch can
select another branch or tag. `pull_request:` and `push:` triggers use the
event branch normally.

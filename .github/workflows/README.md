# CI Workflows

`schedule:` and `workflow_dispatch:` triggers fire **only from the default
branch (`main`)**. A workflow YAML must live on `main` for its cron to
register — the same file on other branches has no effect. `pull_request:`
and `push:` triggers fire from the event branch's file and work normally
on `develop`.

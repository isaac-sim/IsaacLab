# CI Workflows

`schedule:` and `workflow_dispatch:` triggers fire **only from the default
branch (`main`)**. A workflow YAML merged only into `develop` won't register
its cron — it has to be on `main` too. `pull_request:` and `push:` triggers
fire from the event branch's file, so they work normally on `develop`.

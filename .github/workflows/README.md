# CI Workflows

`schedule:` triggers run only from the repository's current default branch.
A scheduled workflow YAML must live on that branch for its cron to register.
A `workflow_dispatch:` workflow must also exist on the default branch,
although manual dispatch can select another branch or tag. `pull_request:`
and `push:` triggers use the event branch normally.

`latest-release-compatibility.yml` runs `main` through the same Docker and test
pipeline as `build.yaml`, but loads the Isaac Sim image pins from the active
`release/3.0.0` branch. Keep its `config_ref` aligned with the active release
branch so the compatibility pipeline follows the released Isaac Sim image
without duplicating its version or digest.

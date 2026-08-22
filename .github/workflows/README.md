# CI Workflows

`schedule:` triggers run only from the repository's current default branch.
A scheduled workflow YAML must live on that branch for its cron to register;
the same schedule on another branch has no effect. A `workflow_dispatch:`
workflow must also exist on the default branch, although a manual dispatch can
select another branch or tag. `pull_request:` and `push:` triggers use the
event branch normally.

## Automated release backports

`backport-release-3.0.yml` handles PRs that merge into `develop` with the
`backport-release-3.0.0` checkbox selected in the PR template. The workflow
must be installed on the repository's default branch because it uses the
trusted `pull_request_target` event. Deploy the workflow, `.github/scripts/backport.py`,
`.github/scripts/resolve_backport_conflicts.py`, and `.github/PULL_REQUEST_TEMPLATE.md`
to that branch together; keep the same files on `develop` so the next
default-branch transition retains the system.

The workflow uses the existing `isaaclab-bot` GitHub App credentials:

- `CHANGELOG_APP_CLIENT_ID`
- `CHANGELOG_APP_PRIVATE_KEY`

The App needs Contents, Pull requests, and Workflows write permission. Clean
cherry-picks are validated for path, patch-ID, and exact edit-content
equivalence before they advance `release/3.0.0`. The push is a normal
fast-forward, so a concurrent release update causes a safe failure instead of
being overwritten.

Conflicting cherry-picks use the same fixed NVIDIA inference endpoint as the
Isaac Lab review bot and require the `NVIDIA_INFERENCE_API_KEY` repository
secret. No OpenAI API key is used. The resolver sends the source-before,
source-after, release, and Git-conflict text to the inference API and accepts
complete UTF-8 contents only for paths Git already marked as conflicted. It
does not receive a GitHub write token. The result is opened as a draft backport
PR and is never pushed directly to the release branch.

The resolver defaults to `azure/openai/gpt-5.6-sol` with
`azure/anthropic/claude-opus-5` as a fallback. Repository variables
`NVIDIA_BACKPORT_MODEL` and `NVIDIA_BACKPORT_FALLBACK_MODEL` can override that
order. Binary, non-UTF-8, oversized, incomplete, or out-of-scope resolutions
fail safely and require a manual backport.

No local polling service is required for backports. GitHub Actions calls the
inference API only when Git reports a conflict.

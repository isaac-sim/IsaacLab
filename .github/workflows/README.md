# CI Workflows

Scheduled workflows run from the repository's default branch. `workflow_run`
and `issue_comment` receivers must also exist there. `workflow_dispatch` must
be enabled on the default branch but can target another branch. `pull_request:`
and `push:` workflows use the event ref and work normally on `develop`.

## Hosted documentation previews

Comment exactly `publish-doc` on an open PR to request hosting. Like `run-ci`,
the command accepts the PR author or a user with write/admin access. It adds the
persistent `docs-preview` label and dispatches the publisher using `GITHUB_TOKEN`;
no additional GitHub App token is needed. Only labeled PRs are hosted, keeping
storage demand limited to requested previews. Removing the label frees that
preview's space without closing the PR. Maintainers can also apply it directly.

`Docs` builds a `docs-html` artifact for PRs and a `docs-release-html` artifact
for nightly or manually dispatched release builds. `Publish Docs` combines the
release site with previews at `/pr-preview/<PR number>/`, deploys to GitHub Pages,
then creates or updates one bot comment and a commit status with the preview URL.
Comments identify the built commit; a failed build leaves the last good preview
available. Closing or merging a PR, or removing its label, removes its preview
and expires its comment. If the current build is running or unavailable, the
request waits for a successful Docs build. Rerun Docs if its artifact has expired.
Reopening a PR restores a preview when a matching build artifact is available.

The publisher runs after `Docs`, on command dispatch, PR closure or preview label
changes, and hourly to recover missed events or notification failures. Every run
reconciles all opted-in open PRs, and one
concurrency group serializes the complete publication. This also handles GitHub
replacing a pending run when several builds finish together. Unchanged sites
are not redeployed. A PR closed during publication is removed by the next run.

The automation-owned `docs-site` branch stores `public/` and a publication
manifest. Only `public/` is uploaded to Pages. Release builds replace the release
tree while preserving open previews; the manifest is saved only after deployment
succeeds (or when no deployment was needed). Bot comment failures are retried.
Removing a preview removes it from the hosted site; generated files remain in
the branch's Git history and in build artifacts until their retention expires.

### Repository setup and rollout

1. Merge the Docs, Publish Docs, and Publish Docs Command workflows and
   `.github/scripts/docs_preview.py` onto the default branch. `workflow_run` and
   `issue_comment` require their receiving workflows there.
2. Keep the existing `REPO_NAME` secret set to the publishing repository's
   `owner/name`. Both release builds and the publisher use it; forks do not
   publish unless explicitly configured.
3. Keep GitHub Pages configured with **GitHub Actions** as its publishing source.
   Allow the `github-pages` environment to deploy from the default branch and
   the maintained PR base branches used by `pull_request_target` close events.
   Environment approval requirements also apply to preview updates and cleanup.
4. Allow `GITHUB_TOKEN` to write the `docs-site` branch, PR comments, commit
   statuses, and Pages deployments. The command also needs Actions write
   permission for explicit dispatch and PR write permission for labels.
   Exempt the generated branch from rules
   requiring PRs or signed commits. No personal access token is needed.
5. Run **Docs** manually on the default branch once to seed the release site.
   The publisher refuses to deploy a preview-only site before this succeeds,
   leaving the existing hosted docs intact. Subsequent nightly builds use the
   same publisher. **Publish Docs** can be run manually to retry publication.

PR builds use a read-only token. The privileged publisher always checks out the
default branch, looks up builds of the known Docs workflow through the GitHub
API, and matches previews to the current PR source repository, branch, and SHA.
It treats artifacts as static files, rejects path traversal and links, and never
executes PR code. Fork builds still follow GitHub's contributor approval policy.
Previews contain contributor-authored HTML and JavaScript and share the Pages
origin with the release documentation. The combined site is capped at 950 MiB
to stay below the [GitHub Pages 1 GB site limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
An oversized site fails before deployment and preserves the last published site.

Run the publisher's tests without simulation dependencies:

```bash
uv run --no-project python -m unittest discover -s .github/scripts -p 'test_docs_preview.py' -v
```

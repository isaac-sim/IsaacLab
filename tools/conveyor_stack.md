# Conveyor integration draft

PR #6952 is the combined validation draft. Review and merge #8326 (meshes),
#8327 (Contrib conveyor support), #8331 (racetrack transfer), and #8332
(warehouse sorting) individually. Do not merge the integration draft or its
temporary tracking tools.

After pushing a change to any component PR, run from this integration checkout:

```bash
uv run --no-project python tools/conveyor_stack.py --check --push
```

To also check the trained policy on CPU:

```bash
uv run --no-project python tools/conveyor_stack.py --check --push \
  --checkpoint /path/to/model.pt
```

The script fetches current `develop` and each current PR head, then merges them
in dependency order in a temporary worktree. Merged PRs are supplied by
`develop` instead of being merged again. Closed, unmerged components require
an explicit change to the list in `conveyor_stack.json`.

Each test file runs in its own process. The refresh covers mesh authoring,
adapter controls/lifecycle, both task contracts, selected batch resets, sorting,
geometry, asset normalization, and the lock file. Optional policy evaluation
requires completed transfers with zero safety resets or non-finite
observations. Formatting also runs before publication.

On success, the script publishes a freshly rebuilt branch with a lease against
concurrent updates and rewrites #6952's description with the exact included
SHAs and completed checks. The same snapshot is committed in
`tools/conveyor_stack.json`. A merge conflict or failed check stops publication.

This runs on demand. Component pushes do not automatically refresh #6952.
GitHub CI runs on the published integration commit; expensive checks can be
requested with the repository's usual `run-ci` comment.

The refresh uses the configured `fork` remote for publication, authenticated
`gh`, and the current checkout's uv environment. Local edits and the current
checkout are preserved; tests use only the published component heads. To
inspect a refreshed result without changing an existing worktree:

```bash
git fetch fork maximiliank/conveyor-franka-env
git worktree add --detach /tmp/conveyor-preview FETCH_HEAD
```

When a prerequisite merges, rebase its dependent drafts onto current
`develop`, update their review-only comparison links, and refresh the
integration again.

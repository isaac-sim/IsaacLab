---
name: isaaclab-bug-fix
description: Fix a reproduced IsaacLab bug by creating a branch, implementing the fix, updating changelogs, running pre-commit, and opening a PR following project guidelines. Use when a bug has been reproduced and needs a code fix, or when asked to fix an issue and create a pull request.
---

# IsaacLab Bug Fix

Implement a fix for a reproduced bug, following all IsaacLab contribution guidelines.

## Inputs

- **Issue number**: GitHub issue `#N`
- **Issue title and description**: For PR context
- **Reproduction steps and error**: From the reproduce phase
- **Affected code location**: Identified during reproduction

## Workflow

### Step 1: Create a feature branch

```bash
git checkout origin/develop
git checkout -b isaaclab-bot/fix-issue-<NUMBER>
```

Branch naming: `isaaclab-bot/fix-issue-<NUMBER>` (e.g. `isaaclab-bot/fix-issue-1234`).

### Step 2: Implement the fix

Follow the coding standards defined in `AGENTS.md` and `docs/source/refs/contributing.rst`. Read both files before implementing.

### Step 3: Write a regression test

Add a test that:
1. **Fails** without the fix (verify by temporarily reverting your change)
2. **Passes** with the fix applied

Use pytest:
```bash
./isaaclab.sh -p -m pytest <PATH_TO_TEST>::<TEST_METHOD>
```

### Step 4: Update the changelog

Determine which `source/<package>/` directories were modified and update each affected package's changelog.

1. Find the current version in `source/<package>/config/extension.toml`
2. Bump the **patch** version (e.g. `1.5.0` → `1.5.1`)
3. Add a new version entry at the top of `source/<package>/docs/CHANGELOG.rst`
4. Update `version` in `source/<package>/config/extension.toml` to match

Changelog entry format:

```rst
X.Y.Z (YYYY-MM-DD)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed <concise description of bug> in :meth:`~package.Class.method`.
  <Brief explanation of the root cause and what was changed.>
```

Use today's date. Use past tense. Use Sphinx cross-references for class/method names.

### Step 5: Run pre-commit

**CRITICAL: Run BEFORE committing.**

```bash
./isaaclab.sh -f
```

If it modifies files, stage them and run again:
```bash
git add -A
./isaaclab.sh -f
```

Repeat until all checks pass.

### Step 6: Commit

```bash
git add -A
git commit -m "$(cat <<'EOF'
Fix #<NUMBER>: <short imperative description>

<What was broken and why. What this commit changes to fix it.
Wrap at 72 characters.>
EOF
)"
```

Rules:
- Imperative mood subject line (~50 chars)
- No trailing period on subject
- Body explains what and why
- **No AI attribution lines**

### Step 7: Push and create PR

```bash
git push -u origin HEAD
```

Create PR using the project template:

```bash
gh pr create --repo isaac-sim/IsaacLab --base develop --title "Fix #<NUMBER>: <short description>" --body "$(cat <<'EOF'
# Description

<Summary of what the bug was and how it's fixed.>

Fixes #<NUMBER>

## Type of change

- Bug fix (non-breaking change which fixes an issue)

## Screenshots

N/A

## Checklist

- [x] I have read and understood the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html)
- [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with `./isaaclab.sh --format`
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] I have updated the changelog and the corresponding version in the extension's `config/extension.toml` file
- [ ] I have added my name to the `CONTRIBUTORS.md` or my name already exists there

EOF
)"
```

### Step 8: Post-PR comment on the issue

```bash
gh issue comment <NUMBER> --repo isaac-sim/IsaacLab --body "A fix has been submitted in PR #<PR_NUMBER>. The root cause was <brief explanation>."
```

## References

- Coding standards and changelog rules: read `AGENTS.md`
- Contributing guidelines and code style: read `docs/source/refs/contributing.rst`
- PR template: read `.github/PULL_REQUEST_TEMPLATE.md`
- Bug report template: read `.github/ISSUE_TEMPLATE/bug.md`

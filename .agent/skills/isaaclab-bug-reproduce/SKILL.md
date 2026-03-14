---
name: isaaclab-bug-reproduce
description: Reproduce a reported bug in IsaacLab by checking out the specified commit, running the reproduction steps, and determining if the bug is still present on the latest commit. Use when asked to reproduce, verify, or test a reported GitHub issue.
---

# IsaacLab Bug Reproduce

Systematically reproduce a reported bug and determine its current status.

## Inputs

- **Issue number**: GitHub issue `#N`
- **Commit hash**: From the issue's System Info or latest `develop`
- **Reproduction steps**: Extracted from the issue body

## Workflow

### Step 1: Checkout the reported commit

Older commits won't have the workflow files. Before switching, save them to a
temp directory and restore them after checkout so the agent context is preserved.

```bash
git fetch origin

# Preserve workflow files that may not exist on the target commit
AGENT_TMPDIR="$(mktemp -d)"
cp -r .agent "$AGENT_TMPDIR/.agent" 2>/dev/null || true
cp AGENTS.md "$AGENT_TMPDIR/AGENTS.md" 2>/dev/null || true
cp CLAUDE.md "$AGENT_TMPDIR/CLAUDE.md" 2>/dev/null || true

git stash  # save any local changes
git checkout <COMMIT_HASH>

# Restore workflow files onto the checked-out tree (don't stage them)
cp -r "$AGENT_TMPDIR/.agent" .agent 2>/dev/null || true
cp "$AGENT_TMPDIR/AGENTS.md" AGENTS.md 2>/dev/null || true
cp "$AGENT_TMPDIR/CLAUDE.md" CLAUDE.md 2>/dev/null || true
```

If the commit hash is invalid or not found, fall back to `origin/develop`:
```bash
git checkout origin/develop

# Restore workflow files (same as above)
cp -r "$AGENT_TMPDIR/.agent" .agent 2>/dev/null || true
cp "$AGENT_TMPDIR/AGENTS.md" AGENTS.md 2>/dev/null || true
cp "$AGENT_TMPDIR/CLAUDE.md" CLAUDE.md 2>/dev/null || true
```

### Step 2: Attempt reproduction

Run the reproduction steps from the issue. Use `./isaaclab.sh -p` to run Python scripts:

```bash
./isaaclab.sh -p <script_or_command>
```

For inline Python:
```bash
./isaaclab.sh -p -c "<python_code>"
```

For tests:
```bash
./isaaclab.sh -p -m pytest <test_path>
```

**Record the output carefully** — capture exit codes, error messages, and stack traces.

### Step 3: Evaluate reproduction result

```
Reproduction attempted
├─ Bug reproduced at reported commit?
│  ├─ YES → Go to Step 4 (test on latest)
│  └─ NO  → Go to Step 5 (comment: cannot reproduce)
```

### Step 4: Test on latest commit

```bash
git checkout origin/develop

# Restore workflow files again after switching
cp -r "$AGENT_TMPDIR/.agent" .agent 2>/dev/null || true
cp "$AGENT_TMPDIR/AGENTS.md" AGENTS.md 2>/dev/null || true
cp "$AGENT_TMPDIR/CLAUDE.md" CLAUDE.md 2>/dev/null || true
```

Re-run the same reproduction steps.

```
Bug on latest develop?
├─ YES (still reproduces) → Route to isaaclab-bug-fix skill
├─ NO  (fixed on latest)  → Go to Step 6 (comment: fixed, close issue)
```

### Step 5: Comment — Cannot Reproduce

Post a comment explaining what was tried and asking for clarification:

```markdown
I attempted to reproduce this issue at commit `<HASH>` with the following steps:

```
<exact commands run>
```

**Result**: The operation completed successfully without the reported error.

<details>
<summary>Full output</summary>

```
<captured output>
```

</details>

Could you please provide more details or a minimal reproducible example? Specifically:
- The exact command or script you ran
- Your full environment details (OS, GPU, CUDA, Isaac Sim version)
- The complete error output

This will help us narrow down environment-specific factors.
```

Post via:
```bash
gh issue comment <NUMBER> --repo isaac-sim/IsaacLab --body "<BODY>"
```

### Step 6: Comment — Fixed on Latest & Close

```markdown
This issue appears to have been resolved on the latest `develop` branch (commit `<LATEST_HASH>`).

I tested with the following steps:

```
<exact commands run>
```

**Result**: The operation completed successfully.

<details>
<summary>Full output</summary>

```
<captured output>
```

</details>

Closing this issue. If you still experience the problem on the latest version, please reopen with updated reproduction steps.
```

Post and close:
```bash
gh issue comment <NUMBER> --repo isaac-sim/IsaacLab --body "<BODY>"
gh issue close <NUMBER> --repo isaac-sim/IsaacLab
```

### Step 7: Clean up

```bash
git checkout develop  # return to develop
git stash pop         # restore any stashed changes (if applicable)
rm -rf "$AGENT_TMPDIR"  # remove temp copy of workflow files
```

## Important Notes

- **Preserve workflow files across checkouts.** `AGENTS.md`, `CLAUDE.md`, and `.agent/` may not exist on older commits. Always copy them to a temp dir before checkout and restore them after. Never stage or commit these restored files.
- Always use `./isaaclab.sh -p` instead of raw `python3` for running scripts
- Capture ALL output (stdout + stderr) for evidence
- If reproduction requires a simulator (Isaac Sim) and it's not available, note this limitation in the comment
- Be precise about which commit was tested — always include the full hash

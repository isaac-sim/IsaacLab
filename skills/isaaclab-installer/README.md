# isaaclab-installer

An [agentskills.io](https://agentskills.io/specification)-compatible skill that helps users install Isaac Lab and Isaac Sim by picking the right combination of options for their use case, hardware, and preferences — and then running it.

## What it does

Isaac Lab supports many installation combinations:

- **Isaac Lab**: from pip OR from source clone
- **Python env**: uv, conda, or stdlib venv
- **Isaac Sim**: from pip, from a pre-built binary, built from source, or skipped entirely (kit-less mode)

Picking the right combination by reading docs is slow, and silent mismatches between driver / GLIBC / Python / Isaac Sim version are the leading cause of broken installs. This skill:

1. Detects system facts (OS, GPU, driver, GLIBC, Python, env tools, disk, RAM).
2. Asks the user about use case + preferences.
3. Recommends the best combo with rationale.
4. Generates a fully-expanded plan (no execution yet).
5. Asks for explicit confirmation.
6. Executes the plan with per-step prompts and structured logs.
7. Runs a headless smoke test.
8. Saves an install profile for reproducibility and easy support.

## Layout

```
isaaclab-installer/
├── SKILL.md                    # The runbook the agent loads.
├── README.md                   # This file.
├── references/                 # On-demand docs (loaded only when needed).
│   ├── compatibility-matrix.md
│   ├── combos-reference.md
│   ├── platform-notes.md
│   ├── troubleshooting.md
│   └── use-case-recipes.md
├── scripts/
│   ├── _lib.py                 # Shared utilities (pure stdlib).
│   ├── _remote.py              # SSH/SFTP wrapper (paramiko, lazy-loaded).
│   ├── preflight.py            # System detector → JSON.
│   ├── recommend.py            # Picks a combo given facts + prefs.
│   ├── plan_install.py         # Resolves a combo into a command list.
│   ├── execute_install.py      # Runs the plan with confirmations + logs.
│   ├── verify.py               # Post-install headless smoke test.
│   ├── doctor.py               # Diagnoses broken installs.
│   └── profile_io.py           # Read / redact / reproduce profile YAML.
└── resources/
    ├── combos.py               # CANONICAL combo data (single source of truth).
    ├── install_profile.template.yaml
    └── smoke_tests/
        └── hello_isaaclab.py
```

## Quick start (manual)

```bash
# 1. Detect system facts
python3 skills/isaaclab-installer/scripts/preflight.py -o /tmp/preflight.json

# 2. Pick a combo (interactive)
python3 skills/isaaclab-installer/scripts/recommend.py \
    --preflight /tmp/preflight.json \
    --output /tmp/rec.json

# 3. Resolve into an executable plan
python3 skills/isaaclab-installer/scripts/plan_install.py \
    --combo $(jq -r .chosen /tmp/rec.json) \
    --preflight /tmp/preflight.json \
    --isaaclab-dir $HOME/IsaacLab \
    --output /tmp/plan.json

# 4. Execute (interactive, with confirmations)
python3 skills/isaaclab-installer/scripts/execute_install.py --plan /tmp/plan.json
```

## Remote (single host)

Add `--remote user@host` to `preflight.py`, `execute_install.py`, `verify.py`, and `doctor.py` to run that step against a remote Linux host. Password-prompted via `getpass`. Requires `pip install --user paramiko` on the workstation (only when using remote mode). See the "Remote Install" section of `SKILL.md` for full usage.

## Design choices

- **Single source of truth**: every combo and its command sequence is defined in `resources/combos.py`. Adding a new combo is one entry; doc pages and CI matrix can both consume it.
- **Zero runtime dependencies**: `preflight.py`, `recommend.py`, `plan_install.py`, `execute_install.py`, and `doctor.py` use only Python's standard library. They run on the user's system Python before any environment is created.
- **Explicit confirmation by default**: every state-changing step pauses unless `--yes` is passed.
- **Structured logs**: every install writes a timestamped log to `~/.isaaclab/logs/` and a profile to `~/.isaaclab/install_profile.yaml`.
- **Telemetry-free analytics**: the profile is local-only; users opt in to share it by attaching a redacted copy to GitHub issues.
- **Linux focus**: v1 supports Linux x86_64 and aarch64. Windows / WSL support is out of scope.

## Adding a new combo

1. Open `resources/combos.py`.
2. Add a new dict to `COMBOS` following the schema documented at the top of that file.
3. Optionally add a smoke-test variant to `resources/smoke_tests/`.
4. Run `python3 scripts/recommend.py --preflight <fixture> --non-interactive --use-case <case>` against fixture preflight files to confirm the recommender picks it for the right scenarios.

## Versioning

The skill itself is at `0.1.0` (see `skill_version` in `install_profile.template.yaml`). The pinned Isaac Sim / PyTorch versions live at the top of `resources/combos.py` and should be bumped when Isaac Lab updates them.

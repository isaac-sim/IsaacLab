---
name: "isaaclab-installer"
description: "Install and set up Isaac Lab and Isaac Sim end-to-end on Linux. Picks the right install combination (pip vs source clone; conda/venv/uv; Isaac Sim binary/pip/OSS-build/kit-less) based on the user's hardware, use case, and preferences. Runs preflight system checks, recommends a combo, asks for explicit confirmation, then auto-installs with per-step checkpoints, structured logging, and a headless smoke test at the end. Also diagnoses broken installs ('doctor mode'), reproduces installs from a saved profile, and generates redacted profiles for sharing in GitHub issues. Use whenever the user wants to install Isaac Lab, install Isaac Sim, set up an Isaac Lab environment, fix a broken Isaac Lab install, reproduce a teammate's setup, or pick which install combination to use."
license: "BSD-3-Clause"
compatibility: "Linux x86_64 / aarch64, Python 3.6+ for the skill scripts (the install itself targets Python 3.12 for Isaac Sim 6.x)."
metadata:
  author: "Krishna Lakhi <klakhi@nvidia.com>"
  tags:
    - isaac-lab
    - isaac-sim
    - installation
    - setup
    - environment
    - troubleshooting
    - doctor
    - reproducibility
  languages:
    - python
    - bash
  frameworks:
    - isaaclab
    - isaacsim
    - newton
  domain: simulation
---

# Isaac Lab Installer

## Overview

This skill installs Isaac Lab end-to-end on Linux. Isaac Lab supports a large matrix of install combinations (Isaac Lab from pip vs source clone × env manager among uv/conda/venv × Isaac Sim from pip/binary/source build/kit-less). Picking the right combo by reading docs alone takes time, and silent mismatches between driver / GLIBC / Python / Isaac Sim version are the most common source of broken installs. This skill removes that.

The skill follows a fixed runbook:

1. **Preflight** — gather system facts (OS, kernel, GPU, driver, GLIBC, Python versions present, conda/uv presence, RAM, disk, existing Isaac installs).
2. **Interview** — ask the user about use case, env-manager preference, and Isaac Sim source preference. Skip questions whose answers are forced.
3. **Recommend** — pick the best combo from `resources/combos.py`, with a rationale and a list of ruled-out alternatives.
4. **Plan** — resolve the chosen combo into a fully expanded ordered command list, no execution yet.
5. **Confirm** — show the user the exact commands. STOP until the user explicitly says yes.
6. **Execute** — run the plan with per-step confirmation, structured logging, sudo prompts, and auth-token prompts as needed.
7. **Verify** — run a headless smoke test. On failure, print a clear, prioritized warning and suggested fixes.
8. **Persist** — write `~/.isaaclab/install_profile.yaml` so the install can be reproduced, diagnosed, or shared.

## When to Use This Skill

- The user says any of: "install Isaac Lab", "set up Isaac Lab", "install Isaac Sim", "help me get started with Isaac Lab", "which install method should I use?", "set up RL training with Isaac Lab", "set up the env for Isaac Lab".
- The user wants to **diagnose** an existing install ("Isaac Lab is broken", "doctor my install", "import isaacsim fails").
- The user wants to **reproduce** another machine's install from a profile file ("set up the same env as <teammate>").
- The user wants to **generate a redacted profile** to attach to an issue.

Do NOT use this skill for:

- Running already-installed Isaac Lab scripts.
- Authoring task / RL environments.
- Building Docker images — for that, the docker/ folder in the repo has its own helper (`./isaaclab.sh -o`).

## Prerequisites

- **OS**: Ubuntu 22.04 LTS or newer is the validated target. Other modern Linux distros usually work.
- **Hardware**: NVIDIA GPU (16 GB+ VRAM recommended), 32 GB RAM, 30 GB disk free.
- **Network**: Outbound HTTPS to github.com, pypi.org, pypi.nvidia.com, download.pytorch.org.
- **Sudo**: required for the `apt-get install` step (system build dependencies). The skill prompts before running any sudo command.

For Isaac Sim 6.x the target Python is 3.12. The skill scripts themselves run on any Python 3.6+ — so you don't need 3.12 installed before starting.

## Step-by-Step Instructions

The skill ships scripts under `<skill-path>/scripts/`. Throughout, `<skill-path>` is the directory containing this `SKILL.md`. When invoked from a Claude Code / Claude Desktop session inside the IsaacLab repo, `<skill-path>` is `skills/isaaclab-installer/`.

### Step 1: Preflight

Run the preflight detector. It writes a JSON document with all system facts.

```bash
python3 <skill-path>/scripts/preflight.py -o /tmp/preflight.json
```

Read the human-readable summary (printed to stderr). If anything is obviously wrong — no GPU, driver missing, broken network — surface that to the user and STOP. The user fixes those before continuing.

### Step 2: Interview the user

Ask:

1. **Use case** — one of: RL research, manipulation/teleop, sim2real, contribute to Isaac Lab, contribute to Isaac Sim, build external extension, kitless / Newton only, just exploring.
2. **Env manager** — uv (default, recommended), conda, venv, or no preference.
3. **Isaac Sim source** — pip (default, requires GLIBC 2.35+), binary download, OSS source build, kit-less (no Isaac Sim), or no preference.
4. **Environment name** — default `env_isaaclab`.
5. **Install location** — default `$HOME/IsaacLab`.

If the user clearly stated some answers in their original prompt, do NOT re-ask those; pass them as CLI flags below.

### Step 3: Recommend a combo

```bash
python3 <skill-path>/scripts/recommend.py \
    --preflight /tmp/preflight.json \
    --non-interactive \
    --use-case <user_use_case> \
    --env-manager <user_env_manager> \
    --isaacsim-source <user_isaacsim_source> \
    --env-name <env_name> \
    --output /tmp/recommendation.json
```

This emits the chosen combo id, the rationale, alternates, and a list of any combos that were ruled out by hard requirements (with reasons). Show the user the chosen combo, the rationale, and any caveats. STOP and ask if they want to proceed, override, or change preferences.

### Step 4: Plan the install

```bash
python3 <skill-path>/scripts/plan_install.py \
    --combo <chosen_combo_id> \
    --preflight /tmp/preflight.json \
    --env-name <env_name> \
    --isaaclab-dir <isaaclab_dir> \
    [--isaacsim-path <isaacsim_path>] \
    --output /tmp/plan.json
```

For binary Isaac Sim combos (`binary-*`), `--isaacsim-path` is required and must point at the **extracted** Isaac Sim directory. If the user has not extracted the binary yet, instruct them: download from <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html> and extract to e.g. `$HOME/isaacsim`. The execute step will pause at a manual checkpoint.

### Step 5: Show the plan and get top-level confirmation

`plan_install.py` already prints a human-readable plan to stderr. Re-display the same plan to the user and explicitly say: **nothing has been changed yet, and execution will pause for confirmation again at every state-changing step.** Then ask for go/no-go.

### Step 6: Execute

```bash
python3 <skill-path>/scripts/execute_install.py --plan /tmp/plan.json
```

This runs interactively. It confirms each step (the user can press Enter to accept), streams output, and writes a timestamped log to `~/.isaaclab/logs/`. For Isaac Sim binary combos it pauses at the "verify Isaac Sim binary is extracted" step until the user confirms the download is done.

If you've already collected explicit consent for ALL steps up front (some workflows require a single approval), pass `--yes` to skip the per-step prompts. Default behavior — and the safer one — is to prompt.

### Step 7: Verify

`execute_install.py` runs the verify step automatically. If you want to re-run verification later:

```bash
python3 <skill-path>/scripts/verify.py --profile ~/.isaaclab/install_profile.yaml
```

Verification runs the smoke test at `<skill-path>/resources/smoke_tests/hello_isaaclab.py` headlessly. If headless fails, it prints a prioritized list of likely causes (driver too old, env not activated, container without GPU, etc.) and the path to the full log. **A headless failure is not necessarily a broken install** — the script tells the user how to retry with a GUI viewport.

### Step 8: Tell the user where their profile is

After a successful install:

```
Your install profile is at: ~/.isaaclab/install_profile.yaml
If you ever file a GitHub issue, run:
    python3 <skill-path>/scripts/profile_io.py redact -o profile-for-issue.yaml
and attach the redacted copy — it gives maintainers the full picture of your setup.
```

## Remote Install (single host, password-based SSH)

The skill can run the same install end-to-end on a remote Linux host over SSH. Scope is intentionally narrow:

- **One target at a time.** No fleet orchestration — for that, hand off `plan.json` to Ansible.
- **Password auth only.** Prompted via `getpass`; never on the CLI, never in logs.
- **One extra dependency.** Remote mode requires `paramiko` on the LOCAL machine (the host you run the skill from). The remote host needs nothing pre-installed beyond standard `python3`, `ssh`, and `sudo`. Install paramiko once with `pip install --user paramiko`. Local installs continue to be zero-dependency.

The flow is identical to a local install — same combos, same plan, same confirmation gates — just with `--remote user@host` added to each script.

```bash
# 1. Preflight on the REMOTE host
python3 scripts/preflight.py --remote user@10.0.0.5 -o /tmp/preflight.json

# 2. Recommend (still local, just reads the preflight JSON)
python3 scripts/recommend.py --preflight /tmp/preflight.json --non-interactive \
    --use-case rl_research --env-manager uv --isaacsim-source any \
    --output /tmp/rec.json

# 3. Plan (still local, but the home + paths come from the remote preflight)
python3 scripts/plan_install.py --combo $(jq -r .chosen /tmp/rec.json) \
    --preflight /tmp/preflight.json \
    --env-name env_isaaclab \
    --isaaclab-dir /home/user/IsaacLab \
    --output /tmp/plan.json

# 4. Execute on the REMOTE host
python3 scripts/execute_install.py --plan /tmp/plan.json --remote user@10.0.0.5
```

The execute step will prompt for **two passwords** at the start: the SSH password, and (if any step needs sudo) the sudo password for the remote user. Both are stored in process memory only. If the remote user has passwordless sudo, press Enter at the sudo prompt.

Manual steps (binary downloads) pause with a clear message that the action must be performed **on the remote host**. The script idles locally until you press Enter.

After a remote install:

- A local `~/.isaaclab/install_profile.yaml` is written on YOUR workstation, with the `remote_target` field populated.
- A copy of the same profile is uploaded to `~/.isaaclab/install_profile.yaml` on the REMOTE host so that future `doctor.py --remote …` / `verify.py --remote …` invocations work without extra arguments.

`verify.py` and `doctor.py` both accept `--remote user@host` so you can re-run them later.

### Security notes for remote mode

- Host key verification uses `AutoAddPolicy` on the first connection (the host fingerprint is added to `~/.ssh/known_hosts` automatically). For sensitive hosts, verify the fingerprint out-of-band first and add it to `known_hosts` yourself before running.
- The SSH password is held in Python memory for the lifetime of the run and is never logged.
- The skill will refuse a `--password` CLI flag — passwords must come from the interactive prompt.

## Doctor Mode (existing install diagnostics)

If the user invokes the skill saying "diagnose my install" or "Isaac Lab is broken":

```bash
python3 <skill-path>/scripts/doctor.py
```

This requires either a saved profile (default `~/.isaaclab/install_profile.yaml`) or `--isaaclab-dir` + `--env-python` flags. It checks:

- Is the repo healthy? (`isaaclab.sh` present, `_isaac_sim` symlink resolves)
- Driver version meets the docs minimum.
- GLIBC version supports pip Isaac Sim (where relevant).
- Env python exists, is 3.12, and can import `isaaclab`, `isaacsim`, `torch`, `numpy`, `omni.client`.

Output is sorted critical → warning → info, each with a suggested fix.

## Reproduce Mode (mirror another machine)

If the user has a `profile.yaml` from a teammate:

```bash
python3 <skill-path>/scripts/profile_io.py --profile teammate-profile.yaml reproduce
```

This prints the exact `plan_install.py` + `execute_install.py` commands to recreate the same install on the current machine. Run preflight first; if requirements don't match (different arch, older driver), the recommender will surface the conflict before any execution.

## Usage Examples

**Example 1: New user, hasn't decided anything yet.**

```
python3 scripts/preflight.py -o /tmp/preflight.json
python3 scripts/recommend.py --preflight /tmp/preflight.json     # interactive interview
# user picks: rl_research, uv, any
# --> chosen: pip-uv-source
python3 scripts/plan_install.py --combo pip-uv-source --preflight /tmp/preflight.json \
    --isaaclab-dir $HOME/IsaacLab --output /tmp/plan.json
python3 scripts/execute_install.py --plan /tmp/plan.json
```

**Example 2: User on Ubuntu 20.04 (GLIBC 2.31).** Recommender automatically rules out all `pip-*` Isaac Sim combos because of GLIBC, picks `binary-uv-source` instead, and prompts for the Isaac Sim binary path.

**Example 3: User is building a downstream extension package.**

```
python3 scripts/recommend.py --preflight /tmp/preflight.json --non-interactive \
    --use-case external_extension --output /tmp/rec.json
# --> chosen: pip-only-uv (no git clone, just pip)
```

**Example 4: User reports "import isaacsim fails".**

```
python3 scripts/doctor.py
# walks the user through the broken thing without modifying anything
```

## Features

- **Single source of truth**: every combo + its ordered command list lives in `resources/combos.py`. Adding a new combo = one edit.
- **Auto-install with explicit confirmation**: state-changing steps always prompt before executing. `--yes` is opt-in.
- **Headless smoke test by default**, with clear, prioritized warning if it fails (and a suggested GUI retry).
- **Auth-token aware**: any step declaring `needs_auth: NGC_API_KEY` triggers a `getpass`-style secret prompt and exports the value only into that step's env. Extend `KNOWN_AUTH_TOKENS` in `scripts/execute_install.py` to add more.
- **Structured logs**: per-install timestamped log under `~/.isaaclab/logs/`. Streamed to the terminal AND saved.
- **Install profile**: `~/.isaaclab/install_profile.yaml` written on success. Doctor / verify / reproduce all read it.
- **Redaction**: `profile_io.py redact` strips home paths and username so the file can be attached to a public issue.
- **Doctor mode**: diagnoses without modifying anything.
- **Reproduce mode**: replays an install from a profile.

## Common Pitfalls

| Symptom                                                                 | Likely cause                                       | Fix                                                                    |
| ----------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'isaacsim'`                        | Env not activated, or pip Isaac Sim wheel skipped | Activate the env; rerun the `install_isaacsim_pip` step.               |
| `GLIBC_2.35 not found`                                                   | Older Linux distro                                 | Recommender will switch you to a `binary-*` combo.                    |
| `import torch` says CUDA unavailable                                     | Wrong PyTorch wheel for arch                       | Re-run install_torch with arch-correct index URL (handled by combos). |
| Headless verify hangs                                                    | Container without `--gpus all`, or no GPU         | Run from a host shell with GPU access.                                 |
| `imgui-bundle` build fails on aarch64                                    | Missing OpenGL/X11 dev headers                     | The combo's apt step adds them — re-run that step.                    |
| `_isaac_sim` symlink dangling                                            | Isaac Sim path moved                               | Re-create with `ln -sfn /new/path _isaac_sim`. Doctor flags this.    |
| Conda env not found by `./isaaclab.sh`                                   | Env not activated in the new shell                 | `conda activate <env_name>` before running.                            |

See `references/troubleshooting.md` for the full list and detailed fixes.

## Configuration

The recommender and combos data have a few user-tweakable knobs:

- **Combo data**: `resources/combos.py` — adding a new install method = one new entry.
- **Pinned versions**: `DEFAULT_ISAACSIM_VERSION`, `DEFAULT_TORCH_PIN`, driver thresholds at the top of the same file.
- **Auth tokens**: `KNOWN_AUTH_TOKENS` dict in `scripts/execute_install.py`.

For deeper detail on every combo's exact command sequence, see `references/combos-reference.md`. For platform-specific quirks (aarch64, WSL2), see `references/platform-notes.md`. For symptom → fix mappings see `references/troubleshooting.md`.

## Related Skills

- `confluence-link-checker` — unrelated, but a good reference for skill structure.

## Additional Resources

- Isaac Lab installation docs: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/>
- Isaac Sim system requirements: <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>
- NVIDIA driver downloads: <https://www.nvidia.com/en-us/drivers/unix/>
- `agentskills.io` skill specification: <https://agentskills.io/specification>

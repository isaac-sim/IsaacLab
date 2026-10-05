# Installing Isaac Lab Evaluations

## Contents

- [Scenario 1: Express Install On Ubuntu 22.04](#scenario-1-express-install-on-ubuntu-2204)
- [Scenario 3: Stated Preferences Override Auto-Pick](#scenario-3-stated-preferences-override-auto-pick)
- [Scenario 4: Preflight Blocker Stops The Flow](#scenario-4-preflight-blocker-stops-the-flow)
- [Scenario 5: Windows 11 Guided Fallback](#scenario-5-windows-11-guided-fallback)
- [Scenario 6: China Asset Region Profile During Installation](#scenario-6-china-asset-region-profile-during-installation)

## Scenario 1: Express Install On Ubuntu 22.04

Query: "Install Isaac Lab."

Expected behavior:

- Runs the read-only preflight detection commands before anything else; no interview questions about use case, env manager, or method.
- Reads the minimums from `docs/source/setup/installation/index.rst` "System requirements", routes to the automatic uv path (`installation-method-uv`), then reads that section and uses its commands verbatim.
- Shows one consolidated plan (system facts, method, exact commands, sudo steps) and asks exactly one go/no-go question.
- Executes all steps unattended after yes, logs to `~/.isaaclab/logs/`, ends with the docs-defined minimal verification command, and tells the user how to activate and use the environment.

Known failure modes:

- Interviews the user (use case, env manager, install directory) instead of auto-picking.
- Asks for confirmation again at each step after the go/no-go.
- Copies install commands or version pins from memory without reading the current install section.
- Skips preflight and starts executing install commands directly.


## Scenario 3: Stated Preferences Override Auto-Pick

Query: "Install Isaac Lab using uv into directory lab3."

Expected behavior:

- Honors uv and `lab3` without re-asking, even if preflight suggests uv.
- Uses the uv workflow of the "Python environment with Isaac Sim" section (`installation-method-python-env`) verbatim.
- Still exactly one go/no-go question, unattended execution, docs verification.

Known failure modes:

- Re-asks the user to choose an env manager the request already specified.
- Ignores the requested environment directory.
- Uses a different env name than the one requested.

## Scenario 4: Preflight Blocker Stops The Flow

Query: "Set up Isaac Lab on this machine" (machine has no NVIDIA GPU).

Expected behavior:

- Preflight detection finds no NVIDIA GPU/driver; no state-changing command runs.
- If detection instead finds existing install artifacts, hands off to `isaaclab-setup-troubleshooting` rather than reinstalling over them.

Known failure modes:

- Proceeds with the automatic uv install and fails midway at the Isaac Sim step.
- Attempts to install an NVIDIA driver unattended.
- Reinstalls over an existing broken environment instead of routing to troubleshooting.

## Scenario 5: Windows 11 Guided Fallback

Query: "Install Isaac Lab on Windows 11."

Expected behavior:

- The non-Linux platform routes to the guided flow: the agent follows the Windows tab in `docs/source/setup/installation/index.rst` — "Automatic setup with uv" (`installation-method-uv`) or "Python environment with Isaac Sim" (`installation-method-python-env`) — interactively.
- Confirms the NVIDIA Windows driver, Python 3.12, and long-path support before prescribing commands.
- Uses the Windows-specific commands from the section rather than paraphrasing shell equivalents.
- Verifies with the docs-defined verification command.

Known failure modes:

- Prescribes bash commands on Windows or paraphrases them into cmd form.
- Skips the long-path support step documented for Windows.
- Applies the Linux express unattended flow where the docs require Windows-specific handling.

## Scenario 6: China Asset Region Profile During Installation

Query: "Install Isaac Lab for a workstation in mainland China."

Expected behavior:

- Runs the normal preflight and chooses the install method from current system requirements rather than changing the
  package source solely because of location.
- Reads the Asset Region Profiles section in `index.rst`, includes the documented China profile setting in the single consolidated plan,
  and handles an existing `ISAACSIM_ASSET_ROOT` according to its documented precedence.
- Runs the normal verification with the profile selected, while stating that this does not prove every mirrored asset
  is available.
- Finds every required full relative path in the current availability manifest before recommending an asset-bearing
  example, treats a missing row as not mirrored, and uses only entries marked `available`.
- Does not use geolocation, persist shell configuration without permission, or expose hard-coded storage endpoints.

Known failure modes:

- Changes package indexes or install methods merely because the workstation is in China.
- Duplicates a release-specific manifest URL, bucket name, or service endpoint in the skill.
- Ignores `ISAACSIM_ASSET_ROOT`, causing the selected profile not to take effect.
- Treats a successful empty-scene launch as proof that all China assets are available.

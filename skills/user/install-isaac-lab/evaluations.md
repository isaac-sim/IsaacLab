# Installing Isaac Lab Evaluations

## Scenario 1: Express Install On Ubuntu 22.04

Query: "Install Isaac Lab."

Expected behavior:

- Runs the read-only preflight detection commands before anything else; no interview questions about use case, env manager, or method.
- Reads the minimums from `docs/source/setup/installation/index.rst` "System requirements", routes to the automatic uv path (`installation-method-uv`), then reads that section's included `uv_run_details.inc` and uses its commands verbatim.
- Shows one consolidated plan (system facts, method, exact commands, sudo steps) and asks exactly one go/no-go question.
- Executes all steps unattended after yes, logs to `~/.isaaclab/logs/`, ends with the docs-defined minimal verification command, and tells the user how to activate and use the environment.

Known failure modes:

- Interviews the user (use case, env manager, install directory) instead of auto-picking.
- Asks for confirmation again at each step after the go/no-go.
- Copies install commands or version pins from memory without reading the current install section.
- Skips preflight and starts executing install commands directly.

## Scenario 2: Older Distro Routed To Downloaded Isaac Sim

Query: "I want to install Isaac Lab on Ubuntu 20.04. My GLIBC is 2.31."

Expected behavior:

- The stated GLIBC is compared against the pip minimum read from `docs/source/setup/installation/index.rst` "System requirements" (not memory), routing to the "Downloaded Isaac Sim package" section (`installation-method-binary`).
- The single confirmation summary calls out the manual Isaac Sim package download as the one manual step.
- Execution pauses only at the download step with the URL from the docs section, then continues unattended.
- Ends with the section's docs-defined verification command (bundled-Python launch plus the Isaac Lab tutorial script from the checkout).

Known failure modes:

- Recommends the automatic uv path anyway and lets the user hit a `GLIBC` version error at install time.
- Copies old GLIBC minimums from memory rather than reading the checkout docs.
- Treats the download pause as a second interview and starts re-asking preferences.

## Scenario 3: Stated Preferences Override Auto-Pick

Query: "Install Isaac Lab using conda into env lab3."

Expected behavior:

- Honors conda and `lab3` without re-asking, even if preflight suggests uv.
- Uses the conda variant of the "Python environment with Isaac Sim" section (`installation-method-python-env`) verbatim.
- Still exactly one go/no-go question, unattended execution, docs verification.

Known failure modes:

- Re-asks the user to choose an env manager the request already specified.
- Silently substitutes uv because it is the docs-Recommended default.
- Uses a different env name than the one requested.

## Scenario 4: Preflight Blocker Stops The Flow

Query: "Set up Isaac Lab on this machine" (machine has no NVIDIA GPU).

Expected behavior:

- Preflight detection finds no NVIDIA GPU/driver; no state-changing command runs.
- Reports the blocker with a fix and offers the documented legacy Newton-only installer (`installation-legacy-installer`) as the no-Isaac-Sim alternative.
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

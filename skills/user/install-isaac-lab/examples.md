# Installing Isaac Lab Examples

## Example 1: Express Install On Ubuntu 22.04 (Default Flow)

User: "Install Isaac Lab."

Suggested agent response:

1. Run the read-only preflight commands from the skill workflow (OS, arch, GLIBC, GPU/driver, env managers, RAM, disk). No blockers; facts point to the automatic uv route.
2. Confirm the minimums against `docs/source/setup/installation/index.rst` "System requirements", then read the "Automatic setup with uv" section (`installation-method-uv`) and its included `uv_run_details.inc` steps verbatim.
3. Show one summary: detected system, chosen method with reasons, exact commands, sudo steps, download size. Ask a single go/no-go.
4. On yes, execute every step unattended, logging to `~/.isaaclab/logs/`, then run the docs-defined verification (`uv run python scripts/tutorials/00_sim/create_empty.py --viz kit`).
5. Hand over: how to activate the env, how to run a first demo, log and profile paths. No other questions asked at any point.

## Example 2: Preflight Routes To Downloaded Isaac Sim On Ubuntu 20.04

User: "Install Isaac Lab on my Ubuntu 20.04 box."

Suggested agent response:

1. Preflight detection reports GLIBC 2.31; the pip minimum read from `docs/source/setup/installation/index.rst` "System requirements" is higher, so route to the downloaded Isaac Sim package.
2. Build the plan from the "Downloaded Isaac Sim package" section (`installation-method-binary`) and its `binaries_details.inc` steps.
3. Single confirmation summary notes the one manual step: downloading and extracting the Isaac Sim package.
4. Execute unattended, pausing only at the download step with the URL from the docs section; resume when the user confirms.
5. Verify with the section's docs-defined verification command (`${ISAACSIM_PATH}/isaac-sim.sh` then the Isaac Lab tutorial script from the checkout).

## Example 3: User Preference Overrides Auto-Pick

User: "Install Isaac Lab with conda, env name lab3."

Suggested agent response:

1. Run preflight as usual. Even if uv is present, the stated preference wins: conda with env name `lab3`, no re-asking.
2. Follow the conda variant of the "Python environment with Isaac Sim" section (`installation-method-python-env`) verbatim, including the `pip_details.inc` steps.
3. Single go/no-go, unattended execution, docs verification, handover.

## Example 4: Preflight Finds A Blocker

User: "Set up Isaac Lab."

Suggested agent response:

1. Preflight detection finds no NVIDIA GPU/driver.
2. Do not install. Report the blocker and its fix line (driver install per `docs/source/setup/installation/index.rst` "System requirements"), and offer the legacy Newton-only installer (`installation-legacy-installer`) as the documented no-Isaac-Sim alternative.
3. Resume the express flow after the user resolves the blocker.

## Example 5: Windows 11 Falls Back To Guided Mode

User: "Install Isaac Lab on Windows 11."

Suggested agent response:

1. The platform is Windows, so use the guided per-tab flow instead of unattended express mode.
2. Follow the Windows tab in `docs/source/setup/installation/index.rst` — "Automatic setup with uv" (`installation-method-uv`) for the docs-Recommended path, or "Python environment with Isaac Sim" (`installation-method-python-env`) if the user manages their own env — interactively: confirm driver, Python 3.12, long-path support, then prescribe the Windows commands verbatim.
3. Keep questions minimal; verify with the docs-defined command.

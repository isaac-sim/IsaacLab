# Isaac Lab 3.1 QA with internal Isaac Sim 6.3

This temporary QA patch uses internal Isaac Sim 6.3 and its standalone asset
importer. Run the commands from your Isaac Lab checkout on the NVIDIA network
or VPN, with uv installed. The patch configures the internal registry.

## Apply the QA patch

Start from the checkout you intend to test, with a clean working tree. Fetch the
original QA commit from the fork and cherry-pick it onto a local QA branch:

```bash
git switch -c qa/3.1-internal-isaacsim
git fetch https://github.com/kellyguo11/IsaacLab-public.git kellyg/release-3.1.0-qa-internal-isaacsim
git cherry-pick 9376446a1c3cd7755d6765133a41c7be8e20861c
```

The PR targets `main` until `release/3.1.0` exists. This patch changes only root
dependency configuration and this README. The committed `uv.lock` stays at the
base branch's versions; refresh it locally before installing or running QA.

## Refresh and install the nightly dependencies

Run this same one-line command on Linux, Windows PowerShell, or Windows Command
Prompt at the start of each QA pass:

```bash
uv sync --extra isaacsim --extra importers -P isaacsim -P isaacsim-asset-isolated -P urdf-usd-converter -P mujoco-usd-converter -P usd-optimize
```

`-P` refreshes package metadata and ignores the corresponding versions in the
existing local lock. Isaac Sim and the standalone importer use
`>=6.3.0.dev0,<6.3.0rc0`, allowing new 6.3 alpha/beta nightlies without editing
the version number. The previous converter and optimizer overrides are removed
so the selected Isaac Sim packages determine their matched dependencies.

For importer-only QA, omit `--extra isaacsim`. To test OV integrations alongside
Isaac Sim, add `--extra ov`; those packages retain the base branch's public pins.

The command updates your local `uv.lock` and installs into `.venv`. Leave the
generated lock uncommitted and attach it to the QA report. Do not use `--frozen`
for the initial refresh: the committed lock still contains the public packages.

If registry requests time out, set `UV_HTTP_TIMEOUT` to `120` before retrying:
`export UV_HTTP_TIMEOUT=120` in Bash, `$env:UV_HTTP_TIMEOUT="120"` in PowerShell,
or `set UV_HTTP_TIMEOUT=120` in Command Prompt.

## Install an already resolved QA package set

To reproduce a report, place its generated `uv.lock` in the repository root and
run this instead of refreshing:

```bash
uv sync --frozen --extra isaacsim --extra importers
```

## Use the documented uv commands

After refreshing once, the normal documented `uv run --extra isaacsim ...`
commands use the locally locked 6.3 packages. Ordinary `uv run` checks whether
the lock matches the project; it does not automatically seek a newer nightly
every run. Add `--frozen` to preserve a recorded lock, and include the same
extras as the installation when you want both integrations present.

Print installed versions without importing or launching Isaac Sim:

```bash
uv run --frozen --extra isaacsim --extra importers python -c "from importlib.metadata import version; packages = ('isaacsim', 'isaacsim-asset-isolated', 'isaacsim-robot-schema', 'urdf-usd-converter', 'mujoco-usd-converter', 'usd-optimize'); print({p: version(p) for p in packages})"
```

## Verified package set

The full Linux x86_64 install dry-run on November 20, 2026, resolved 445 packages,
including:

| Package | Version |
| --- | --- |
| Isaac Sim and standalone importer | `6.3.0a43+release.58948.6d33ca13.gl` |
| URDF USD converter | `0.4.0` |
| MuJoCo USD converter | `0.9.0` |
| USD Optimize | `1.4.0` |

Later nightly builds may specify different converter versions. Record the
installed-version output and the generated lock with every QA result.

To discard only the locally generated lock after recording the result:

```bash
git restore uv.lock
```

Refresh again before the next QA run. GitHub-hosted jobs cannot access the
internal index; frozen jobs using the committed lock exercise the public package
set rather than this QA resolution.

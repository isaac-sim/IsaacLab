# Isaac Lab 3.1 QA with internal Isaac Sim 6.3

This temporary QA patch uses internal Isaac Sim 6.3 with the public standalone
`isaacsim-asset-isolated==6.2.0.1` importer. Run the commands from your Isaac Lab
checkout on the NVIDIA network or VPN, with uv installed. The patch configures
the internal registry for Isaac Sim.

## Apply the QA patch

Start with a clean working tree and create a local branch from the QA fork:

```bash
git fetch https://github.com/kellyguo11/IsaacLab-public.git kellyg/release-3.1.0-qa-internal-isaacsim
git switch -c qa/3.1-internal-isaacsim FETCH_HEAD
```

To apply only the QA changes to another base, use the patch commands listed in
[the draft PR](https://github.com/isaac-sim/IsaacLab/pull/9137). The PR targets
`main` until `release/3.1.0` exists. The QA changes affect only root dependency
configuration and this README. The committed `uv.lock` stays at the base
branch's versions; refresh it locally before installing or running QA.

## Refresh and install the nightly dependencies

Run this same one-line command on Linux, Windows PowerShell, or Windows Command
Prompt at the start of each QA pass:

```bash
uv sync --extra isaacsim --extra importers -P isaacsim
```

`-P isaacsim` refreshes Isaac Sim and its version-matched runtime packages,
ignoring the previous runtime version in the local lock. The QA override uses
`>=6.3.0rc3,<6.4`, allowing new 6.3 release candidates without editing build
numbers. Keep `-P isaacsim` in the nightly refresh command; the range alone does
not upgrade an existing lock.

The standalone importer remains pinned to public `6.2.0.1`. The converter and
USD compatibility overrides match the base branch. Isaac Sim retains its own
matching `isaacsim-asset==6.3...` Kit bundle. There is no standalone asset 7.0
override and no internal standalone importer dependency.

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
commands use the locally locked 6.3 runtime and public 6.2 standalone importer.
Ordinary `uv run` checks whether the lock matches the project; it does not
automatically seek a newer nightly every run. Add `--frozen` to preserve a
recorded lock, and include the same extras as the installation when you want
both integrations present.

When Kit is running, Isaac Lab enables and imports the URDF/MJCF extensions from
Isaac Sim's asset bundle. Without Kit, it imports the public standalone importer.
The public importer does not select or replace the Kit extensions.

Print installed versions without importing or launching Isaac Sim:

```bash
uv run --frozen --extra isaacsim --extra importers python -c "from importlib.metadata import version; packages = ('isaacsim', 'isaacsim-asset', 'isaacsim-asset-isolated', 'isaacsim-robot-schema', 'urdf-usd-converter', 'mujoco-usd-converter', 'usd-optimize'); print({p: version(p) for p in packages})"
```

## Verified package set

The combined Linux x86_64 installation on November 25, 2026, resolved 451
packages. URDF and MJCF conversions passed both with Kit running and without
Kit. The dependency-configuration tests passed (13 tests), as did the repository
formatting and lint checks. The installed versions included:

| Package | Version |
| --- | --- |
| Isaac Sim and Kit asset bundle | `6.3.0rc4+release.59505.a720a5f8.gl` |
| Public standalone importer | `6.2.0.1` |
| Robot schema | `6.2.0.0` |
| URDF USD converter | `0.3.4` |
| MuJoCo USD converter | `0.7.0` |
| USD Optimize | `1.3.1` |

The nightly install dry-run also resolved Isaac Sim
`6.3.0rc5+release.59636.188cdb4a.gl` with the same public importer. Runtime smoke
tests used the cached RC4 build shown above. Record the installed-version output
and the generated lock with every QA result.

To discard only the locally generated lock after recording the result:

```bash
git restore uv.lock
```

Refresh again before the next QA run. GitHub-hosted jobs cannot access the
internal index; frozen jobs using the committed lock exercise the public package
set rather than this QA resolution.

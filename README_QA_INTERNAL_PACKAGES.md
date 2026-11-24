# Isaac Lab 3.1 QA with internal Isaac Sim 6.3

This temporary QA patch uses internal Isaac Sim 6.3 with the latest standalone
`isaacsim-asset` package, including 7.0 builds. Run the commands from your Isaac
Lab checkout on the NVIDIA network or VPN, with uv installed. The patch
configures the internal registry.

## Apply the QA patch

Start with a clean working tree and create a local branch from the QA fork:

```bash
git fetch https://github.com/kellyguo11/IsaacLab-public.git kellyg/release-3.1.0-qa-internal-isaacsim
git switch -c qa/3.1-internal-isaacsim FETCH_HEAD
```

For cherry-picking only the QA changes onto another base, use the commits listed
in [the draft PR](https://github.com/isaac-sim/IsaacLab/pull/9137). The PR targets
`main` until `release/3.1.0` exists. The QA changes affect only root dependency
configuration and this README. The committed `uv.lock` stays at the base
branch's versions; refresh it locally before installing or running QA.

## Refresh and install the nightly dependencies

Run this same one-line command on Linux, Windows PowerShell, or Windows Command
Prompt at the start of each QA pass:

```bash
uv sync --extra isaacsim --extra importers -P isaacsim -P isaacsim-asset -P urdf-usd-converter -P mujoco-usd-converter -P usd-optimize
```

`-P` refreshes package metadata and ignores the corresponding versions in the
existing local lock. Isaac Sim uses `>=6.3.0rc3,<6.4`, including new release
candidates. The asset requirement and UV override use `>=6.3.0.dev0,<8`, allowing
both 6.3 and 7.x builds. The override replaces Isaac Sim's exact 6.3 asset
requirement; widening the ordinary requirement alone would still conflict with
that pin. With `-P isaacsim-asset`, UV selects the newest available asset build
in the range, currently 7.0, independently of the runtime's asset pin.

The range alone does not upgrade an existing local lock: keep `-P` in the nightly
refresh command. `isaacsim-asset-isolated` is no longer requested. The selected
asset package supplies its matched converter, common, schema, and optimizer
dependencies.

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
commands use the locally locked 6.3 runtime and 7.0 asset packages. Ordinary
`uv run` checks whether the lock matches the project; it does not automatically
seek a newer nightly every run. Add `--frozen` to preserve a recorded lock, and
include the same extras as the installation when you want both integrations
present.

When Kit is running, Isaac Lab enables and imports the Kit URDF/MJCF extensions.
Without Kit, it imports the standalone asset wheel. The pip override selects the
installed asset distribution; it does not override Kit's extension selection.
In the combined smoke test, Kit downloaded its importer extensions from NVIDIA's
extension registry on first use. Allow access to that registry as well as the
internal pip registry when testing Kit conversions.

Print installed versions without importing or launching Isaac Sim:

```bash
uv run --frozen --extra isaacsim --extra importers python -c "from importlib.metadata import version; packages = ('isaacsim', 'isaacsim-asset', 'isaacsim-common', 'isaacsim-robot-schema', 'urdf-usd-converter', 'mujoco-usd-converter', 'usd-optimize'); print({p: version(p) for p in packages})"
```

## Verified package set

The combined Linux x86_64 installation on November 24, 2026, resolved 449
packages. URDF and MJCF conversions passed both with Kit running and without
Kit. The dependency-configuration tests passed (13 tests), as did the repository
formatting and lint checks. The installed versions included:

| Package | Version |
| --- | --- |
| Isaac Sim | `6.3.0rc4+release.59505.a720a5f8.gl` |
| Asset, common, and robot schema | `7.0.0a2.dev58610+gl.698c657` |
| URDF USD converter | `0.4.0` |
| MuJoCo USD converter | `0.9.0` |
| USD Optimize | `1.3.1` |

Later nightly builds may specify different converter versions. Record the
installed-version output and the generated lock with every QA result.

To discard only the locally generated lock after recording the result:

```bash
git restore uv.lock
```

Refresh again before the next QA run. GitHub-hosted jobs cannot access the
internal index; frozen jobs using the committed lock exercise the public package
set rather than this QA resolution.

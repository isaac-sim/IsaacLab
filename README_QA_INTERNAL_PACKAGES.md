# Isaac Lab 3.1 internal package QA

This branch tests the new standalone Isaac Sim 7 asset package and the latest
Isaac Sim 6.3 runtime candidates. Run the commands from the repository root on
the NVIDIA network or VPN. The branch configures the required registry.

## Apply the QA patch

From a clean checkout, create a local QA branch and apply the QA commits:

```bash
git switch -c qa/3.1-internal-packages
git fetch https://github.com/kellyguo11/IsaacLab-public.git kellyg/release-3.1.0-qa-internal-isaacsim
git cherry-pick 15077020d6..FETCH_HEAD
```

If the original QA commit `9376446a1c` is already applied, fetch the branch as
above and cherry-pick only the new follow-up commit shown in the PR.

The committed `uv.lock` stays identical to the PR's base. Each QA pass generates
a local lock; leave it uncommitted and attach it to the QA report.

## Choose one package set

The new standalone `isaacsim-asset` package replaces `isaacsim-asset-isolated`.
Full Isaac Sim 6.3 also depends on a distribution named `isaacsim-asset`, but
requires its own matching 6.3 build. Therefore `--extra importers` and
`--extra isaacsim` are mutually exclusive on this QA branch. UV records separate
resolutions and rejects commands that enable both extras.

Use separate checkout environments when switching repeatedly between these
workflows. A normal sync in one checkout replaces packages from the other set.

### Standalone asset importer QA: latest 7.0 nightly

The same one-line command works in Bash, PowerShell, and Command Prompt:

```bash
uv sync --extra importers -P isaacsim-asset -P isaacsim-common -P isaacsim-robot-schema -P urdf-usd-converter -P mujoco-usd-converter -P usd-optimize
```

The `importers` extra requires `isaacsim-asset>=7.0.0.dev0,<7.0.0rc0` from the
internal index. The selected asset package determines its matching common,
schema, converter, and optimizer dependencies. Add other extras such as `test`,
`video`, or `ov` as needed.

### Full Isaac Sim QA: latest compatible 6.3 candidate

```bash
uv sync --extra isaacsim -P isaacsim -P isaacsim-asset -P isaacsim-robot-schema -P urdf-usd-converter -P mujoco-usd-converter -P usd-optimize
```

The runtime override is `isaacsim[all,extscache]>=6.3.0rc3,<6.4`. This includes
RC3 and later 6.3 builds, with the matching 6.3 asset distribution. This workflow
does not test the standalone 7.0 asset package.

## Refresh versus reproduce

`-P` refreshes registry metadata and replaces the selected package versions in
the local lock. Run the appropriate refresh command at the start of each QA
pass. Do not use `--frozen` before this initial refresh.

To reproduce a report, place its generated `uv.lock` at the repository root and
install the same package set without refreshing:

```bash
uv sync --frozen --extra importers
```

For a full-runtime report, use `--extra isaacsim` instead. The documented
`uv run` commands work with the selected extra and locally refreshed lock.
Add `--frozen` when a test must preserve the recorded resolution.

Print the standalone importer versions without launching a runtime:

```bash
uv run --frozen --extra importers python -c "from importlib.metadata import version; packages = ('isaacsim-asset', 'isaacsim-common', 'isaacsim-robot-schema', 'urdf-usd-converter', 'mujoco-usd-converter', 'usd-optimize'); print({p: version(p) for p in packages})"
```

If registry requests time out, set `UV_HTTP_TIMEOUT` to `120` before retrying:
`export UV_HTTP_TIMEOUT=120` in Bash, `$env:UV_HTTP_TIMEOUT="120"` in PowerShell,
or `set UV_HTTP_TIMEOUT=120` in Command Prompt.

## Package versions checked on November 23, 2026

| Package | Standalone importer QA | Full runtime QA |
| --- | --- | --- |
| Isaac Sim | Not installed | `6.3.0rc3+release.59333.3164eac4.gl` |
| Isaac Sim asset | `7.0.0a2.dev58610+gl.698c657` | Same 6.3 build as the runtime |
| Common and robot schema | Same 7.0 build as the asset package | Selected by the runtime |
| URDF USD converter | `0.4.0` | Selected by the runtime |
| MuJoCo USD converter | `0.9.0` | Selected by the runtime |
| USD Optimize | `1.3.1` | Selected by the runtime |

The standalone set was installed on Linux x86_64 and exercised through Isaac
Lab's URDF and MJCF converters. Both produced USD geometry, joints, articulation
roots, and reusable cached outputs without launching Kit. The importer emitted
warnings for unsupported MJCF procedural textures/lights and missing actuator
gain data in the fixtures. Full Isaac Sim RC3 was validated by dependency
resolution only; Windows and ARM64 runtime execution were not tested locally.

Record installed versions and the generated lock for every QA result, since
later nightly builds may select different dependencies. After recording the
result, `git restore uv.lock` restores the committed public-package lock; refresh
again before the next QA pass.

GitHub-hosted runners cannot access the internal registry. Frozen jobs using
the committed lock exercise the public package set rather than these QA builds.

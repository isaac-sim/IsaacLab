# Release 3.0.0 Isaac Sim Internal Package QA

This branch prepares Isaac Lab `release/3.0.0` for QA with unreleased Isaac Sim
packages from NVIDIA's internal Python registry. OVPhysX, OVRTX, and OVStage are
not overridden and use the versions published by the release branch. This is a
temporary QA branch and is not intended to merge while the internal package
source is required.

Run these commands from the repository root on the NVIDIA network or VPN. The
branch already configures the required index; do not add an index URL to the
commands below.

## Check out the QA branch

```bash
git fetch https://github.com/kellyguo11/IsaacLab-public.git \
  kellyg/release-3.0.0-qa-internal-packages
git switch --detach FETCH_HEAD
```

## Package selection

The root UV overrides use a bounded nightly range instead of changing the
release branch's published Isaac Sim pins. The committed `uv.lock` intentionally
remains identical to `release/3.0.0` to keep this branch small and conflict-free.
Each QA pass must generate a local lock before running tests.

The Linux x86_64 resolution below was last verified on October 10, 2026.

| Package | QA override | Registry | Last verified resolution |
| --- | --- | --- | --- |
| Isaac Sim | `>=6.2.0.dev0,<6.2.0rc0` | Internal Isaac Sim | `6.2.0a21+develop.52446.b91212b0.gl` |
| Isaac Sim asset importer | `>=6.2.0.dev0,<6.2.0rc0` | Internal Isaac Sim | `6.2.0a21+develop.52446.b91212b0.gl` |
| OVPhysX, OVRTX, and OVStage | unchanged | Public | Per `release/3.0.0` |

## QA regression to retest

Retest [NVBug 6833897](https://nvbugspro.nvidia.com/bug/6833897) with the
refreshed local resolution and attach the version output from this README to the
QA result. This README does not claim that the package refresh fixes the bug.

## Start a QA pass with the latest nightly

Use the package upgrade selectors when starting a new QA pass. They make UV
ignore the Isaac Sim versions already recorded in `uv.lock` and select the
newest versions that satisfy the QA overrides above:

```bash
UV_HTTP_TIMEOUT=120 uv sync \
  --extra isaacsim \
  --extra importers \
  -P isaacsim \
  -P isaacsim-asset-isolated
```

This initial refresh is required. Do not use `--frozen` before running it. The
command updates the local `uv.lock`; leave that file uncommitted. Record the
resolved versions and attach the generated lock to the QA report because
separate passes can select different nightly builds as packages are published.

## Reproduce a reported QA resolution

Download the `uv.lock` attached to the QA report, place it at the repository
root, and then install without allowing UV to change it:

```bash
UV_HTTP_TIMEOUT=120 uv sync --frozen \
  --extra isaacsim \
  --extra importers
```

`--frozen` reproduces the exact versions and hashes in the attached lock rather
than looking for newly published builds.

## Verify installed versions

Print the exact Isaac Sim versions and include them in the QA report:

```bash
uv run --frozen \
  --extra isaacsim \
  --extra importers \
  python -c 'from importlib.metadata import version; packages = ("isaacsim", "isaacsim-asset-isolated", "isaacsim-robot-schema"); print({p: version(p) for p in packages})'
```

## Use the documented `uv run` commands

The `uv run` commands in the Isaac Lab documentation work the same way on this
branch. UV uses the versions in the locally refreshed `uv.lock`:

```bash
uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct physics=isaacsim_physx
```

After the initial refresh, add `--frozen` immediately after `uv run` when a test
must not update the lock:

```bash
uv run --frozen --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct physics=isaacsim_physx
```

Do not add `-P` to every documented command. Refresh once at the start of the QA
pass, record the versions, and then use ordinary or frozen `uv run` commands for
the entire pass.

## Return to the shared snapshot

After the QA pass, discard the locally generated lock to return to a clean
checkout:

```bash
git restore uv.lock
```

The restored lock contains the public release packages, not the QA nightly.
`uv lock --check` is therefore expected to report that the committed lock needs
an update on this branch. GitHub-hosted CI cannot resolve the NVIDIA-internal PDX
registry hostname, so non-frozen dependency jobs cannot exercise this QA setup.

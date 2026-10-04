# Release 3.0.0 Internal Package QA

This branch prepares Isaac Lab `release/3.0.0` for QA with unreleased OVRTX and
OVStage packages from NVIDIA's internal Python registries. Isaac Sim is not
overridden and keeps the release branch's published pin. It is a
temporary QA branch and is not intended to merge while those internal package
sources are required.

Run these commands from the repository root on the NVIDIA network or VPN.
The branch already configures the required indexes; do not add index URLs to the
commands below.

## Check out the QA branch

```bash
git fetch https://github.com/kellyguo11/IsaacLab-public.git \
  kellyg/release-3.0.0-qa-internal-packages
git switch --detach FETCH_HEAD
```

## Package selection

The root UV overrides use release-compatible ranges instead of changing the
release branch's published package pins. The committed `uv.lock` intentionally
remains identical to `release/3.0.0` to keep this branch small and conflict-free.
Each QA pass must generate a local lock before running tests.

The Linux x86_64 resolution below was last verified on October 1, 2026.

| Package | QA override | Registry | Last verified resolution |
| --- | --- | --- | --- |
| OVRTX | `>=0.5.1,<0.6` | Internal Omniverse | `0.5.1.385274` |
| OVStage | `>=0.2.1,<0.3` | Internal Omniverse | `0.2.1.385274` |
| OVPhysX | unchanged at `==0.6.3` | Public PyPI | `0.6.3` |
| Isaac Sim and asset importer | unchanged (release pin) | Public | Per `release/3.0.0` |

The UV override for OVStage intentionally replaces OVPhysX 0.6.3's dependency
on the older `ovstage==0.2.0.377349` build.

## QA regression to retest

Retest [NVBug 6833897](https://nvbugspro.nvidia.com/bug/6833897) with the
refreshed local resolution and attach the version output from this README to the QA
result. This README does not claim that the package refresh fixes the bug.

## Start a QA pass with the latest nightlies

Use the package upgrade selectors when starting a new QA pass. They make UV
ignore the versions already recorded for these packages and select the newest
versions that satisfy the QA overrides above:

```bash
UV_HTTP_TIMEOUT=120 uv sync \
  --extra isaacsim \
  --extra importers \
  --extra ov \
  -P ovrtx \
  -P ovstage
```

This initial refresh is required. Do not use `--frozen` before running it. The
command updates the local `uv.lock`; leave that file uncommitted. Record the
resolved versions and attach the generated lock to the QA report because
separate passes can select different builds as new packages are published.

## Reproduce a reported QA resolution

Download the `uv.lock` attached to the QA report, place it at the repository
root, and then install without allowing UV to change it:

```bash
UV_HTTP_TIMEOUT=120 uv sync --frozen \
  --extra isaacsim \
  --extra importers \
  --extra ov
```

`--frozen` reproduces the exact versions and hashes in the attached lock rather
than looking for newly published builds.

## Verify installed versions

After installing all three QA extras, print the exact versions and include them
in the QA report:

```bash
uv run --frozen \
  --extra isaacsim \
  --extra importers \
  --extra ov \
  python -c 'from importlib.metadata import version; packages = ("isaacsim", "isaacsim-asset-isolated", "ovphysx", "ovrtx", "ovstage"); print({p: version(p) for p in packages})'
```

## Use the documented `uv run` commands

The `uv run` commands in the Isaac Lab documentation work the same way on this
branch. The extras route packages to the configured internal or public registry,
and UV uses the versions in the locally refreshed `uv.lock`:

```bash
# OVPhysX and OVStage
uv run --extra ovphysx isaaclab train --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct physics=ovphysx

# OVRTX, OVPhysX, and OVStage
uv run --extra ov isaaclab train --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct physics=ovphysx

# Full Isaac Sim
uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct physics=isaacsim_physx

# Multiple integrations in one environment
uv run --extra isaacsim,importers,ov isaaclab train --rl_library rsl_rl \
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

The restored lock contains the public release packages, not the QA nightlies.
`uv lock --check` is therefore expected to report that the committed lock needs
an update on this branch. GitHub-hosted CI cannot resolve the NVIDIA-internal PDX
registry hostname, so non-frozen dependency jobs cannot exercise this QA setup.
Also note that `uv pip check` can report the original release pins; the
project-level UV overrides deliberately replace that metadata during resolution.

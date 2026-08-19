<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# CI wheelhouse profiles

The CI wheelhouse is a profile-specific, binary-only cache. Profiles with
`mirror_lock = true` mirror every compatible registry wheel in `uv.lock` from
its exact locked URL and check its locked SHA-256 digest. They do not resolve
locked packages again. Selection honors Linux architecture resolution markers
and the lock's `requires-python`, bounds manylinux tags to profile-supported
maximum glibc floor (including intermediate and legacy floors), and keeps one
preferred artifact per package/version (exact CPython, then ABI3, then pure
Python, with newer numeric wheel build tags breaking equal-compatibility ties).

The checked-in profiles cover focused and full-lock hosted CPython 3.12,
Isaac Sim CPython 3.12 on x86_64 and aarch64, and version-parameterized
compatibility jobs. The focused hosted profile resolves CI and documentation
roots without mirroring the simulator graph. License, wheel-resolution, and
kit-less jobs use it as an offline-first bootstrap and retain online fallback
for their larger environment-specific graphs. Compatibility profiles require
`--base_version`; that value scopes the cache to the matching Isaac Sim image.

The hosted profile sets `mirror_lock = false` because singleton hosted jobs need
only their CI tooling, not the full Isaac Sim dependency graph. In that mode the
builder skips lock-wheel selection and passes every profile `ci_root` to
target-aware `pip download --only-binary=:all:` so the output contains those
roots and their binary dependency closure. Isaac Sim and compatibility profiles
keep `mirror_lock = true`. The resolved value is part of the manifest profile
and therefore its profile hash.

Roots-only profiles may put packages whose runtime dependencies are already
provided by the consumer image in `ci_roots_no_deps`. The builder downloads
these roots in a separate target-aware `pip download --no-deps` invocation and
records them as `roots.pip_no_deps` in the manifest. The runtime profile uses
this for `leapp`, avoiding its CUDA and PyTorch closure, while normal
`ci_roots` retain dependency resolution. Full-lock profiles continue to mirror
the locked `leapp` and OV wheels.

The common runtime artifact intentionally omits OVRTX and OVPhysX. OVRTX is
source-only at the pinned version, while the binary OV wheels are large and
used by only a small test subset. Those jobs retain the existing pinned
wheelhouse-image or online fallback path instead of sending multi-gigabyte OV
artifacts to every test shard.

Root downloads use a temporary constraints file generated from unambiguous,
target-compatible registry versions in `uv.lock`. Its entry count and SHA-256
are recorded in the manifest so otherwise-floating roots and dependencies stay
deterministic where the lock provides a usable version.

Full-lock profiles set `exclude_package_prefixes = ["isaacsim"]`. Matching
registry packages, including Isaac Sim extension-cache distributions, are
recorded as intentional `profile-excluded` manifest entries instead of being
downloaded. Consumers rely on the Isaac Sim base image for these packages and
retain an online fallback if an excluded package is unexpectedly absent.

CI-root downloads are isolated in a temporary staging directory. Successfully
downloaded wheels are atomically moved into `wheelhouse/`, including usable
partial results from a failed pip invocation. Before inventorying, the builder
removes wheels that are neither selected lock artifacts nor part of the current
staging result. This keeps broad-prefix cache restores from retaining stale
versions while preserving exact locked-wheel cache hits.

Build a profile from the repository root:

```bash
uv run --no-project python -m tools.ci_wheelhouse build \
    --profile isaacsim-cp312-x86_64 \
    --output_dir .ci-wheelhouse/isaacsim-cp312-x86_64
```

The output contains `wheelhouse/`, `manifest.json`, and, only for a successful
build, a `complete` sentinel. `manifest.json` is deterministic for identical
inputs and wheel contents. It records input hashes, the resolved profile,
package versions, file hashes, CI-root handling, exclusions, and partial-build
errors.

Use `verify` before consuming a restored cache:

```bash
uv run --no-project python -m tools.ci_wheelhouse verify \
    --output_dir .ci-wheelhouse/isaacsim-cp312-x86_64 \
    --lock_file uv.lock \
    --profiles_file .github/ci-wheelhouse/profiles.toml
```

## Intentional exclusions

The wheelhouse never builds source artifacts:

- `editable-source`: repository workspace packages remain editable installs.
- `git-source`: VCS dependencies retain their existing online install path.
- `virtual-source`: the root workspace project is metadata, not an artifact.
- `sdist-only`: a registry package has an sdist but no locked wheel.
- `incompatible-wheel-tags`: locked wheels exist, but none match the profile's
  CPython tag, ABI, manylinux architecture, or pure-`any` tags.
- `profile-excluded`: the normalized package name matches a profile exclusion
  prefix and is expected to come from the consumer base image.
- `no-wheel-artifact` and `non-registry-source`: malformed or unsupported lock
  entries that are outside the binary wheelhouse contract.

These entries are documented in the manifest instead of being passed to a
wheel builder. Downstream jobs may keep an explicit online fallback for them.
MoviePy 1.x, used by the record-video lane, is one such source-only package;
its wheel-capable dependencies remain cached while MoviePy uses that fallback.

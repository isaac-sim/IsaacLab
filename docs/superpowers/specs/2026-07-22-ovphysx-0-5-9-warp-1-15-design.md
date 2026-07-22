# OVPhysX 0.5.9 and Warp 1.15 Dependency Update Design

## Goal

Adopt the public OVPhysX 0.5 release line and stable Warp 1.15 while keeping
Isaac Lab's source installs, built wheel metadata, and compatibility CI aligned.
Migrate the OVPhysX manager from the removed file-loading API to the OVPhysX
0.5.9 ovstage attachment API, and verify whether the temporary USD file can be
eliminated safely.

## Dependency Policy

- Declare the OVPhysX optional dependency as `ovphysx>=0.5,<0.6` in
  `source/isaaclab_ovphysx/pyproject.toml`.
- Require exactly `warp-lang==1.15.0` in the Isaac Lab source package and both
  mirrored wheel-builder dependency lists.
- Run daily compatibility coverage against the concrete public release
  `ovphysx==0.5.9`.
- Do not commit `uv.lock`. The repository does not track it; uv resolution is a
  verification step that must resolve OVPhysX 0.5.9 and Warp 1.15.0.

## OVPhysX Runtime Migration

OVPhysX 0.5.9 no longer exposes `PhysX.add_usd()`. Its supported Python API
creates a caller-owned `ovstage.Stage`, populates that stage, and attaches it
with `PhysX.attach_ovstage()`. `OvPhysxManager` will retain the ovstage object
for as long as PhysX is attached and release resources in lifetime-safe order.

Warmup will:

1. Configure the existing live USD physics-scene prim.
2. Produce the env-0-scoped USD input used by the OVPhysX clone fast path.
3. Construct or reset the `PhysX` instance as today.
4. Create an `ovstage.Stage`, populate its physics domain at the initial sealed
   ordinal, and attach it to PhysX.
5. Replay pending clones, warm GPU buffers when applicable, and initialize the
   existing tensor-backed scene-data adapter.

Reset and close will invalidate tensor bindings before detaching or resetting
the attached ovstage. The manager will destroy its owned ovstage only after
OVPhysX no longer references it. Existing `_stage_path` behavior will remain
available because articulation tendon-name recovery currently reads the
filtered USD stage independently of OVPhysX.

## In-Memory Stage Investigation

The installed 0.5.9 wheel will be inspected and exercised for a public,
supported population API that can consume either the existing `pxr.Usd.Stage`,
an anonymous `Sdf.Layer`, or serialized USD bytes. The acceptance criteria for
removing the temporary file are all of the following:

- no dependency on private or undocumented ovstage symbols;
- no mutation of Isaac Lab's live stage;
- preservation of env-0 filtering before OVPhysX clone replay;
- correct resolution of referenced assets and schema data;
- passing the same CPU OVPhysX scene-loading smoke test.

The published 0.5.9 documentation currently exposes
`ovstage.population.open_usd(stage, path, ...)` and describes a distinct,
namespaced OpenUSD runtime. Therefore, the default implementation retains the
temporary env-0-scoped USDA file as the supported bridge. The implementation
will remove it only if the installed public API demonstrably meets every
criterion above. Directly translating the complete Isaac Lab USD stage into
ovstage is outside this PR's scope.

## Tests and Verification

Regression coverage will be test-first:

- Extend `source/isaaclab/test/cli/test_wheel_builder_metadata.py` to require
  the exact Warp 1.15.0 pin and the OVPhysX 0.5 dependency range.
- Add focused OVPhysX manager lifecycle tests for ovstage population,
  attachment, reset, and cleanup order using small fakes around the external
  runtime boundary.
- Confirm each new regression test fails against the pre-change declarations
  or manager integration before implementing the corresponding change.
- Resolve dependencies with uv and assert the selected distributions are
  `ovphysx==0.5.9` and `warp-lang==1.15.0`.
- Install the public wheel and run a CPU OVPhysX runtime/scene-loading smoke
  test that exercises a real stage, rather than only testing imports.
- Run the focused unit suites, then `./isaaclab.sh -f` on all files.

If the real-wheel smoke test reveals additional OVPhysX 0.5.9 API changes in
the stage-loading lifecycle, those compatibility changes are in scope when
they are covered by a focused regression test. Unrelated simulation behavior
or feature work remains out of scope.

## Changelog and PR Scope

Add one patch changelog fragment for `isaaclab` describing the stable Warp
1.15.0 pin and one for `isaaclab_ovphysx` describing the OVPhysX 0.5 range and
ovstage migration. Existing compiled changelogs and `config/extension.toml`
files will not be edited.

The PR will contain the dependency declarations, mirrored wheel metadata,
daily compatibility input, required manager migration, regression tests, and
the two changelog fragments. It will not add dependencies beyond those already
provided transitively by the OVPhysX distribution.

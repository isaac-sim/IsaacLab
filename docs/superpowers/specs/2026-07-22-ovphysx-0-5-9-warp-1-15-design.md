# OVPhysX 0.5.9 and Warp 1.15 Dependency Update Design

## Goal

Adopt the public OVPhysX 0.5 release line and stable Warp 1.15 while keeping
Isaac Lab's source installs, built wheel metadata, and compatibility CI aligned.
Migrate the OVPhysX manager from the removed file-loading API to the OVPhysX
0.5.9 ovstage attachment API, and verify whether the temporary USD file can be
eliminated safely.

## Dependency Policy

- Declare the OVPhysX optional dependency as `ovphysx>=0.5,<0.6` in the root
  `pyproject.toml`, which is the repository's dependency source of truth.
- Require exactly `warp-lang==1.15.0` in the same root dependency table.
- Remove the aarch64 OVPhysX 0.4.13 CI override now that 0.5.9 publishes an
  aarch64 wheel.
- Update the tracked `uv.lock`; resolution must select OVPhysX 0.5.9 and Warp
  1.15.0.

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
OVPhysX no longer references it. Articulation tendon-name recovery will parse
the retained in-memory USDA instead of reopening a stage file.

## In-Memory Stage Investigation

The installed 0.5.9 wheel will be inspected and exercised for a public,
supported population API that can consume serialized USD. The acceptance
criteria for removing the temporary file are all of the following:

- no dependency on private or undocumented ovstage symbols;
- no mutation of Isaac Lab's live stage;
- preservation of env-0 filtering before OVPhysX clone replay;
- correct resolution of referenced assets and schema data;
- passing the same CPU OVPhysX scene-loading smoke test.

OVStage 0.1.0 exposes
`ovstage.population.open_usd_from_string(stage, usda, ...)`. A real runtime
probe confirmed that it can populate the physics domain and attach to OVPhysX
0.5.9. The manager will therefore flatten the composed live stage to an
anonymous layer, remove cloned environments other than env 0 from that layer,
serialize it to USDA text, and populate OVStage directly from the string. The
live Isaac Lab stage remains unmodified and no temporary file is required.

## OmniClient Compatibility Gate

OVPhysX 0.5.9's paired OVStage runtime declares OmniClient
`2.72.3-release.7151+gl.5390bed9`. Isaac Lab currently pins the publicly
available `omniverseclient==2.72.1`. Runtime probes show that loading 2.72.1
first makes OVPhysX fail closed because the versions differ, while loading the
OVStage runtime first and then importing the 2.72.1 Python binding segfaults.

The final dependency set therefore requires `omniverseclient==2.72.3`. At
implementation time NVIDIA's public package index does not yet list that
wheel, so lock regeneration and the end-to-end Cartpole smoke remain gated on
its publication. The PR must not claim runtime compatibility while 2.72.1 is
resolved.

## Tests and Verification

Regression coverage will be test-first:

- Extend `source/isaaclab/test/cli/test_uv_run_pyproject.py` to require the
  exact Warp 1.15.0 pin and the OVPhysX 0.5 dependency range.
- Add focused OVPhysX manager lifecycle tests for ovstage population,
  attachment, reset, and cleanup order using small fakes around the external
  runtime boundary.
- Confirm each new regression test fails against the pre-change declarations
  or manager integration before implementing the corresponding change.
- Resolve dependencies with uv and assert the selected distributions are
  `ovphysx==0.5.9` and `warp-lang==1.15.0`.
- Install the public wheel and run a CPU OVPhysX runtime/scene-loading smoke
  test that exercises a real stage, rather than only testing imports.
- Run the focused unit suites and a Cartpole OVPhysX smoke, then
  `./isaaclab.sh -f` on all files.

If the real-wheel smoke test reveals additional OVPhysX 0.5.9 API changes in
the stage-loading lifecycle, those compatibility changes are in scope when
they are covered by a focused regression test. Unrelated simulation behavior
or feature work remains out of scope.

## Changelog and PR Scope

Add one patch changelog fragment for `isaaclab` describing the stable Warp
1.15.0 pin and one for `isaaclab_ovphysx` describing the OVPhysX 0.5 range and
ovstage migration. Existing compiled changelogs and `config/extension.toml`
files will not be edited.

The PR will contain the root dependency declarations, tracked lock update, CI
input, required manager migration, regression tests, and the two changelog
fragments. It will not add dependencies beyond the matching OmniClient runtime
required by OVPhysX 0.5.9.

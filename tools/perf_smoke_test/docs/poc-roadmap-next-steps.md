# Perf Smoke POC Roadmap: Next Steps

This note tracks two known follow-up items for the POC. They are not blockers for
the current NVCR-backed smoke gate, but they should be addressed before treating
the system as fully general across historical dependency eras and automated
root-cause workflows.

## 1. Automatic Era Image Selection

Current state:

- The smoke gate pulls `nvcr.io/nvidian/isaac-lab:latest-perf` by default.
- Published images are also tagged immutably as `sha-<short>`.
- Baseline comparison already filters FPS samples by runtime/config metadata, but
  the workflow does not automatically choose an older image for older eras.
- Older-era testing currently requires manually setting `PERF_SMOKE_CI_IMAGE` to
  the correct immutable tag.

Roadmap:

- Publish an image manifest that maps runtime eras to immutable NVCR tags.
- Have the gate compute or infer the required runtime era before pulling an image.
- Resolve `runtime_contract_hash -> nvcr.io/...:sha-<short>` automatically.
- Keep `PERF_SMOKE_CI_IMAGE` as a manual override/debug escape hatch.

This makes historical replay and bisection more robust without changing the
current-branch PR path, where `latest-perf` is the expected production image.

## 2. Integrate The Bisection Agent With The Current Gate

Current state:

- The bisection agent was developed against the older `perf_smoke_test`
  layout.
- The active POC framework has been renamed/refactored to `perf_smoke_test` and
  `perf-smoke-*` workflows.
- The bisection code can still run from an internally consistent demo branch, but
  it should not be merged while it still depends on stale paths/imports.

Roadmap:

- Port bisection modules, scripts, docs, and workflow references onto
  `tools/perf_smoke_test`.
- Update imports, artifact names, and CLI references from `perf_smoke_test`
  / `perf-smoke` to `perf_smoke_test` / `perf-smoke`.
- Reuse the existing smoke-gate oracle instead of adding a second detector.
- Make bisection consume the same benchmark artifacts, baseline metadata, and
  image-resolution path as the PR gate.
- After automatic era image selection exists, let each bisected candidate resolve
  the correct NVCR image for its dependency era.

Short term, the bisection demo can remain isolated on its own branch. Long term,
the bisection agent should become a first-class consumer of the current smoke
gate, not a parallel copy of the old framework.

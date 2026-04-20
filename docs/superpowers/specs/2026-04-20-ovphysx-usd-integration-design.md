# OVPhysX USD Integration Improvement — Design Spec

**Issue:** [#5324 — \[OVPHYSX\] Improve USD integration to avoid disk dumps](https://github.com/isaac-sim/IsaacLab/issues/5324)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Currently, `OvPhysxManager` exports the USD stage to a temporary file on disk and then loads it back into the ovphysx PhysX instance. This disk round-trip is unnecessary overhead and introduces temp file management complexity. The goal is to pass the USD stage directly to ovphysx without writing to disk.

## Current Problem

From `OvPhysxManager._warmup_and_load()`:
1. The in-memory USD stage is exported to a temp `.usda` file via `stage.Export()`
2. ovphysx loads the file via `physx.add_usd(temp_path)`
3. The temp file persists until process exit

Issues:
- **Performance:** Disk I/O is slow for large scenes (many environments, complex assets)
- **Reliability:** Temp file may not be cleaned up on crash
- **Containerization:** Some container environments have limited temp storage
- **Security:** USD file on disk may contain sensitive scene data

## Desired State

Pass the USD stage directly to ovphysx via an in-memory API, eliminating the disk round-trip entirely.

## Approach Options

### Option A: In-memory USD string (ovphysx API addition)

Add an ovphysx API that accepts a USD string/bytes instead of a file path:

```python
# Current
physx.add_usd("/tmp/exported_scene.usda")

# Proposed
usd_string = stage.ExportToString()
physx.add_usd_from_string(usd_string)
```

Or using flatbuffers/binary format:
```python
usd_bytes = stage.Flatten().ExportToString()  # or binary format
physx.add_usd_from_bytes(usd_bytes)
```

### Option B: Shared memory / file descriptor

Use a file descriptor or shared memory segment instead of a named file:
```python
import io
buffer = io.BytesIO()
stage.Export(buffer)  # if pxr supports stream export
physx.add_usd_from_fd(buffer.fileno())
```

**Risk:** USD's `Export()` may only support file paths, not streams.

### Option C: Direct scene graph transfer

Instead of going through USD serialization at all, transfer the scene graph directly:
- Iterate prims in the USD stage
- Call ovphysx APIs to create physics objects directly
- Skip serialization entirely

**Risk:** High complexity, essentially reimplementing USD loading in Python.

## Recommendation

**Option A (in-memory USD string)** is the cleanest. It requires @marcodiiga to add `add_usd_from_string()` or similar to the ovphysx Python API. The IsaacLab side change is minimal (replace `stage.Export(path)` + `physx.add_usd(path)` with `stage.ExportToString()` + `physx.add_usd_from_string(string)`).

**Blocker for @marcodiiga:** Add an in-memory USD loading API to ovphysx. The C++ side likely already parses USD from a buffer — this just needs a Python binding that passes the buffer directly instead of reading from a file path.

## Implementation Steps

1. @marcodiiga adds `physx.add_usd_from_string()` to ovphysx wheel
2. Update `OvPhysxManager._warmup_and_load()`:
   - Replace `stage.Export(tmp_path)` with `stage.ExportToString()`
   - Replace `physx.add_usd(tmp_path)` with `physx.add_usd_from_string(usd_string)`
   - Remove temp file creation and cleanup
3. Remove the `atexit` temp file cleanup handler

## Dependencies

- **Requires ovphysx API addition** (`add_usd_from_string` or equivalent)
- **Requires PR #4852 merged**
- **Independent of all other specs**

## Estimated Scope

- IsaacLab-side changes: ~10 lines in `ovphysx_manager.py`
- ovphysx-side: new Python binding for in-memory USD loading

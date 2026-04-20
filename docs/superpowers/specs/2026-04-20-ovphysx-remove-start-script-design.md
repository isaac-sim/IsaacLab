# OVPhysX Remove Start Script — Design Spec

**Issue:** [#5323 — \[OVPHYSX\] Remove start script](https://github.com/isaac-sim/IsaacLab/issues/5323)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Remove or replace the `scripts/run_ovphysx.sh` start script. Currently, OVPhysX requires a separate shell script to launch because it needs to override `LD_PRELOAD` to load its bundled PhysX libraries instead of Kit's. The goal is to make OVPhysX usable through the standard `./isaaclab.sh -p` launcher.

## Current Problem

From PR #4852 review:
- `run_ovphysx.sh` unconditionally overwrites `LD_PRELOAD` rather than prepending, silently discarding any existing preloaded libraries
- The script cannot coexist with Kit (it overrides Kit's library paths)
- Users must remember to use a different launch script for OVPhysX vs PhysX

## Desired State

OVPhysX should be launchable via the standard `./isaaclab.sh -p` command, with the physics backend selected by configuration (e.g., `physics_cfg=OvPhysxCfg()`), not by which shell script you run.

## Approach Options

### Option A: Integrate LD_PRELOAD into OvPhysxManager initialization

Instead of a shell wrapper, have `OvPhysxManager.initialize()` programmatically adjust library loading:
- Use `ctypes.CDLL` to preload the required ovphysx shared libraries at Python runtime
- Or use `os.environ["LD_PRELOAD"]` early in the import chain (before any conflicting libraries are loaded)

**Risk:** `LD_PRELOAD` must be set before the process starts for it to affect library resolution. Setting it after process start may not work for all libraries.

### Option B: Use a Python entry point that re-execs with correct LD_PRELOAD

`OvPhysxCfg` detection in the launcher causes a `os.execve` re-launch with the correct `LD_PRELOAD`:
- First invocation detects OVPhysX config, sets `LD_PRELOAD`, re-execs
- Second invocation runs normally with correct libraries

**Risk:** Adds complexity to the launcher. Double-start may confuse debuggers.

### Option C: Namespace-isolate ovphysx's Carbonite (ovphysx-side fix)

Have ovphysx bundle its libraries with a different soname so they don't conflict with Kit's:
- No `LD_PRELOAD` needed at all
- Both Kit and ovphysx can coexist in the same process

**This is the ideal long-term solution** but requires work from @marcodiiga on the ovphysx packaging side.

## Recommendation

**Short term:** Option B (re-exec with correct `LD_PRELOAD` via launcher detection). This is pragmatic and works now.

**Long term:** Option C (namespace-isolated Carbonite in ovphysx). This eliminates the problem entirely.

**Blocker for @marcodiiga:** The dual-Carbonite conflict (`os._exit(0)` workaround in `OvPhysxManager`) and the `LD_PRELOAD` requirement are both symptoms of the same root cause — ovphysx and Kit share the same Carbonite library. Namespace isolation on the ovphysx side would fix both issues.

## Implementation Steps

1. Move `LD_PRELOAD` logic from `run_ovphysx.sh` into `isaaclab.sh` or the Python launcher
2. Detect OVPhysX backend selection early in the startup path
3. If OVPhysX is selected and `LD_PRELOAD` is not set, re-exec with correct value
4. Remove `run_ovphysx.sh` after migration

## Dependencies

- **Requires PR #4852 merged**
- **Option C requires ovphysx packaging changes**
- **Independent of asset/sensor specs**

## Estimated Scope

- Launcher integration: ~50 lines in `isaaclab.sh` or `sim_launcher.py`
- Testing: manual verification of launch flow
- Script removal: delete `scripts/run_ovphysx.sh`

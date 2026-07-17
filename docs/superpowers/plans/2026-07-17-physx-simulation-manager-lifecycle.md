# PhysX SimulationManager Lifecycle Hotfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Isaac Sim's original `SimulationManager` callbacks from invalidating Isaac Lab PhysX tensor views when `isaaclab_physx` is imported before Kit startup.

**Architecture:** Keep the existing lazy compatibility hook for safe pre-Kit imports, and guarantee a second invocation from `PhysxManager.initialize()` after Kit exists. Verify the import-order contract with an app-owning integration test that observes the original manager before PhysX initialization.

**Tech Stack:** Python 3.12, Isaac Sim Kit, PhysX tensor API, pytest, RST changelog fragments.

## Global Constraints

- Base the branch on current `origin/develop` and keep PR #6337 unchanged.
- Do not restore Kit extension manifests or add dependencies.
- Keep the compatibility hook private, idempotent, and safe before Kit startup.
- Follow Isaac Lab Google-style documentation and PEP 8 conventions.
- Verify the regression test fails before the production fix and passes afterward.
- Run `./isaaclab.sh -f` before every commit and before pushing.

---

### Task 1: Guarantee PhysX lifecycle ownership after Kit startup

**Files:**
- Create: `source/isaaclab_physx/test/sim/test_simulation_manager_ownership.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/__init__.py:50-58`
- Modify: `source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py:281-296`
- Create: `source/isaaclab_physx/changelog.d/antoiner-physx-simulation-manager-lifecycle.rst`

**Interfaces:**
- Consumes: private `isaaclab_physx._patch_isaacsim_simulation_manager()`, `PhysxManager.initialize(sim_context)`.
- Produces: guaranteed post-Kit lifecycle takeover whenever the PhysX backend initializes; no new public API.

- [ ] **Step 1: Write the failing import-order regression test**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify PhysX lifecycle ownership when its package is imported before Kit starts."""

# Import the PhysX config before AppLauncher to reproduce config resolution in normal entry points.
from isaaclab_physx.physics import PhysxCfg

from isaaclab.app import AppLauncher
from isaaclab.test.utils import resolve_test_sim_device

simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

import pytest

import isaacsim.core.simulation_manager as simulation_manager_module
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxManager

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_teardown():
    """Create a fresh stage and simulation context for each test."""
    SimulationContext.clear_instance()
    sim_utils.create_new_stage()
    yield
    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
def test_initialize_claims_simulation_manager_lifecycle():
    """PhysxManager initialization disables Isaac Sim's original lifecycle callbacks."""
    original_manager = simulation_manager_module.SimulationManager
    assert original_manager is not PhysxManager

    sim = SimulationContext(cfg=SimulationCfg(physics=PhysxCfg()))

    assert sim.physics_manager is PhysxManager
    assert simulation_manager_module.SimulationManager is PhysxManager
    assert not any(original_manager.get_default_callback_status().values())
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
env PYTHONPATH=source/isaaclab_physx:source/isaaclab_newton:source/isaaclab_ov:source/isaaclab_ovphysx:source/isaaclab_visualizers \
  ./isaaclab.sh -p -m pytest \
  source/isaaclab_physx/test/sim/test_simulation_manager_ownership.py -q
```

Expected: FAIL at `simulation_manager_module.SimulationManager is PhysxManager`, proving the original manager still owns callbacks after `SimulationContext` initialization.

- [ ] **Step 3: Add the minimal post-Kit retry**

At the start of `PhysxManager.initialize()` add the circular-import-safe local import and invocation:

```python
    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the physics manager."""
        from isaaclab_physx import _patch_isaacsim_simulation_manager

        _patch_isaacsim_simulation_manager()

        from isaaclab.sim.utils.stage import get_current_stage_id
```

Update `_patch_isaacsim_simulation_manager()` documentation to state that package import is opportunistic and `PhysxManager.initialize()` guarantees the retry after Kit startup.

- [ ] **Step 4: Run the regression test to verify GREEN**

Run the Step 2 command again.

Expected: `1 passed`; the module alias points to `PhysxManager` and every default callback on the retained original class is disabled.

- [ ] **Step 5: Add the package changelog fragment**

```rst
Fixed
^^^^^

* Fixed PhysX tensor views being invalidated when the PhysX backend configuration
  was imported before Kit startup.
```

- [ ] **Step 6: Run focused unit and integration verification**

```bash
env PYTHONPATH=source/isaaclab_physx \
  ./isaaclab.sh -p -m pytest \
  source/isaaclab_physx/test/renderers/test_isaac_rtx_renderer_cfg.py -q

env PYTHONPATH=source/isaaclab_physx:source/isaaclab_tasks:source/isaaclab_assets:source/isaaclab_rl:source/isaaclab_newton:source/isaaclab_visualizers \
  ./isaaclab.sh -p scripts/benchmarks/startup.py \
  --task Isaac-Reach-Franka --num_envs 16 --viz none \
  --benchmark_formatter json --output_path /tmp/franka-physx-lifecycle-hotfix
```

Expected: focused unit test passes; Franka completes environment creation, reset, and first step without invalidated-view warnings or errors.

- [ ] **Step 7: Run changelog and repository gates**

```bash
./isaaclab.sh -p tools/changelog/cli.py check develop
./isaaclab.sh -f
git diff --check
```

Expected: every command exits zero and pre-commit reports all hooks passed.

- [ ] **Step 8: Commit the implementation**

```bash
git add \
  source/isaaclab_physx/isaaclab_physx/__init__.py \
  source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py \
  source/isaaclab_physx/test/sim/test_simulation_manager_ownership.py \
  source/isaaclab_physx/changelog.d/antoiner-physx-simulation-manager-lifecycle.rst
git commit -m "Fix PhysX simulation lifecycle ownership" -m \
  "Retry the SimulationManager compatibility takeover when PhysxManager initializes so early backend config imports cannot leave Isaac Sim callbacks subscribed."
```

- [ ] **Step 9: Prepare and open the pull request**

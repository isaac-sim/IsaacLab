# Test Overhaul Examples

Each case is drawn from the `isaaclab_newton` overhaul, which is the reference suite.

## Contents

- [Split a mixed module](#split-a-mixed-module)
- [Fold a standalone test into an existing scene](#fold-a-standalone-test-into-an-existing-scene)
- [Collapse a parametrize into one scene](#collapse-a-parametrize-into-one-scene)
- [Trim a device cross product](#trim-a-device-cross-product)
- [Cover independent axes pairwise](#cover-independent-axes-pairwise)
- [Declare an optional extra](#declare-an-optional-extra)

## Split a mixed module

`physics/test_newton_fabric_body_sync.py` launched Kit at module scope but held ~130 lines of
pure-Python fakes exercising `NewtonManager._initialize_fabric_body_prims`, which needs no
application at all.

The fakes and their tests moved verbatim to `physics/test_newton_fabric_body_sync_unit.py`,
importing only `from isaaclab_newton.physics import NewtonManager`. The original kept the real
Fabric test and gained the marker:

```python
import pytest
from isaaclab.app import AppLauncher

pytestmark = pytest.mark.requires_kit

simulation_app = AppLauncher(headless=True, enable_cameras=True).app
```

`import pytest` is hoisted above the `AppLauncher` import and `pytestmark` is set before the
`.app` line, so the marker is present in a module whose body cannot run without Kit.

## Fold a standalone test into an existing scene

`TestDelayedPDAuthoring` and three sibling classes each ran a full simulation purely to inspect
how actuators had been authored. `_run_simulation` was taught to capture that during a run
already happening:

```python
actuator_info = []
if use_newton_actuators:
    for actuator in SimulationManager.get_model().actuators:
        actuator_info.append({
            "controller_type": type(actuator.controller).__name__,
            "has_delay": actuator.delay is not None,
        })
```

Four classes then collapsed into one method each on the existing equivalence classes:

```python
class TestDelayedPDEquivalence(_EquivalenceTestBase):
    def test_newton_authoring_uses_pd_controller_with_delay(self):
        """Author delayed PD actuators as a PD controller carrying a delay model."""
        for actuator in self.newton_result["actuator_info"]:
            self.assertEqual(actuator["controller_type"], "ControllerPD")
            self.assertTrue(actuator["has_delay"])
```

## Collapse a parametrize into one scene

`test_collision_decimation_invokes_mid_loop_collide` parametrized five decimation values, building
five scenes to vary one cheap attribute. It became one scene and a loop, restoring the attribute:

```python
original = NewtonManager._collision_decimation
try:
    for decimation in (1, 2, 3, 4, 5):
        NewtonManager._collision_decimation = decimation
        ...
finally:
    NewtonManager._collision_decimation = original
```

## Trim a device cross product

Behavior tests do not need every instance count on every device. The initialization tests keep one
CPU and one CUDA case through a paired parametrize; everything else uses the primary device:

```python
# Prefer CUDA for repeated behavior checks when available. The initialization
# cases below retain CPU and CUDA coverage.
PRIMARY_DEVICE = test_devices()[-1:]


@pytest.mark.parametrize(
    ("num_articulations", "device"),
    [(1, test_devices()[0]), (2, test_devices()[-1])],
)
def test_initialization_floating_base_non_root(...):
    ...
```

Write the rationale next to the constant. A reviewer must be able to see which coverage was
deliberately dropped.

## Cover independent axes pairwise

Shape, contact pipeline, and device are independent, so the full matrix builds four times the
scenes it needs. Two complementary case sets cover every value of every axis between them:

```python
def _shape_pipeline_device_cases(complement: bool):
    """Build complementary pairwise cases across shape, pipeline, and device."""
    devices = test_devices()
    cases = []
    for index, shape_type in enumerate(STABLE_SHAPES):
        use_mujoco_contacts = bool(index % 2)
        device = devices[0] if index % 4 in (2, 3) else devices[-1]
        if complement:
            use_mujoco_contacts = not use_mujoco_contacts
            device = devices[-1] if device == devices[0] else devices[0]
        cases.append(pytest.param(device, use_mujoco_contacts, shape_type))
    return cases
```

## Declare an optional extra

A module that imports an optional dependency states it rather than skipping silently:

```python
import pytest

pytestmark = pytest.mark.requires_extra("ovphysx")
```

Running without the extra fails the session with the exact install command. Do not reach for
`pytest.importorskip`: a silent skip reports green while covering nothing.

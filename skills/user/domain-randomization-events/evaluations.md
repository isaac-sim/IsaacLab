# Event Randomization Evaluations

## Contents

- Scenario 1: reset randomization
- Scenario 2: prestartup-only randomization
- Scenario 3: interval disturbance
- Scenario 4: PhysX and Newton material randomization
- Scenario 5: direct workflow events
- Scenario 6: success-driven ADR

## Scenario 1: Reset Randomization

Query: "Randomize initial joint positions every episode."

Expected behavior:

- Selects reset mode.
- Places randomization in event terms.
- Checks entity names, ranges, and reset behavior.

Known failure modes:

- Adds randomization to the reward function.
- Uses startup mode for per-episode behavior.

## Scenario 2: Prestartup-Only Randomization

Query: "Randomize authored USD properties for each run, but I also want them to change every reset."

Expected behavior:

- Identifies that some USD-stage or topology-level randomizations are prestartup-only.
- Warns not to place prestartup-only changes in reset or interval events.
- Suggests pre-generated variants, separate authored assets, or fixed-per-run randomization if per-reset changes are unsupported.

Known failure modes:

- Promises per-reset changes for a property that must be authored before simulation starts.
- Omits the timing limitation from the plan.

## Scenario 3: Interval Disturbance

Query: "Apply random pushes to the robot during training."

Expected behavior:

- Selects interval mode.
- Defines magnitude and interval ranges.
- Recommends small-scale rollout validation before training at scale.

Known failure modes:

- Implements pushes in the training loop.
- Does not check tensor device and shape compatibility.

## Scenario 4: PhysX and Newton Material Randomization

Query: "Randomize friction and restitution for both PhysX and Newton."

Expected behavior:

- Checks the current event implementation for backend-specific behavior.
- Uses PhysX bucket behavior for PhysX.
- Calls out that Newton uses one friction coefficient and ignores `dynamic_friction_range` and `num_buckets`.
- Recommends separate `PresetCfg` event configs and separate smoke tests per backend.

Known failure modes:

- Copies PhysX material parameters directly into Newton.
- Ignores CPU/GPU differences in the event implementation.

## Scenario 5: Direct Workflow Events

Query: "Add domain randomization events to my DirectRLEnv task."

Expected behavior:

- Recognizes that direct RL environments can use `EventManager` through the `events` config field.
- Keeps observations and rewards in direct workflow methods.
- Adds event terms only for randomization behavior.
- References the direct workflow randomization tutorial.

Known failure modes:

- Claims event terms are only for manager-based environments.
- Converts the task to manager-based solely to use randomization events.

## Scenario 6: Success-Driven ADR

Query: "Start gravity at zero and automatically ramp it to -9.81 as my lift policy succeeds."

Expected behavior:

- Keeps gravity application in a reset event.
- Uses a task-owned success-driven scheduler and `modify_term_cfg` interpolation in a manager-based curriculum.
- Places the scheduler before the gravity interpolation term.
- Validates the fixed final range, both interpolation endpoints, promotion/demotion behavior, and clamping.
- References the curriculum guide and Core Lift ADR example.

Known failure modes:

- Treats ADR as an event mode or modifies gravity in the training loop.
- Adds a `curriculum` field to a direct environment and assumes it will execute.
- Imports Core Lift's private scheduler into an unrelated task.
- Expands an unvalidated final range or omits endpoint checks.

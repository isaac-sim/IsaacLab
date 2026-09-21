# Environment Building Evaluations

## Scenario 1: Choose A Workflow

Query: "I want to build a quadruped locomotion task with custom command sampling and custom rewards."

Expected behavior:

- Asks for assets, actions, observations, rewards, reset conditions, sensors, and backend targets.
- Recommends manager-based workflow first and maps custom commands and rewards to reusable command and reward terms.
- Points to current maintained locomotion examples such as `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`, robot-specific configs under `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/`, and manager-based environment tutorials.
- Mentions direct workflow only if the user needs bespoke step/reset logic that cannot fit managers.
- Defines a smoke-test plan before training.

Known failure modes:

- Defaults to direct workflow just because the request mentions custom rewards or commands.
- Copies tutorial code without checking the task's action and observation requirements.
- Uses stale source paths from older Isaac Lab layouts instead of current `source/isaaclab_tasks/isaaclab_tasks/core/` or `contrib/` examples.

## Scenario 2: Build A Manager-Based Task

Query: "Create a reusable reaching task where observations and rewards should be reusable across robot arms."

Expected behavior:

- Recommends manager-based workflow because observations and rewards are reusable.
- Maps task pieces to scene, observation, action, command, reward, termination, event, and curriculum configs.
- Points to manager-based tutorials and source examples.

Known failure modes:

- Implements all behavior in a direct environment even though reuse is the user's main requirement.
- Adds custom abstractions before checking existing MDP terms.

## Scenario 3: Register And Train

Query: "I created a new environment config and need to train it with Gym registration."

Expected behavior:

- Points to the Gym registration and RL training tutorials.
- Verifies the environment imports, resets, and steps before training.
- Checks that the agent config matches the selected training framework.

Known failure modes:

- Starts a large training run before checking reset and step behavior.
- Mixes agent config formats from different RL frameworks.

## Scenario 4: Route A Targeted Sensor Change

Query: "Add foot contact observations and air-time rewards to my quadruped task."

Expected behavior:

- Routes to `isaaclab-using-sensors-actuators` instead of loading this skill as the primary workflow.
- Treats the request as a targeted change to an existing task rather than scaffolding or restructuring an environment.
- Inspects the task's existing contact sensor and MDP terms before proposing new shared code.

Known failure modes:

- Claims the request because it modifies an existing task example.
- Creates a new shared-core observations module without first configuring the sensor and checking existing terms.

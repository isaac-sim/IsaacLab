# Environment Building Evaluations

## Scenario 1: Choose A Workflow

Query: "I want to build a quadruped locomotion task with custom command sampling and custom rewards."

Expected behavior:

- Asks for assets, actions, observations, rewards, reset conditions, sensors, and backend targets.
- Recommends direct workflow first if custom control flow is central to the task.
- Points to maintained locomotion examples and direct environment tutorials.
- Defines a smoke-test plan before training.

Known failure modes:

- Starts from manager terms without preserving the requested custom task behavior.
- Copies tutorial code without checking the task's action and observation requirements.

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

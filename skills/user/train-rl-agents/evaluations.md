# RL Training Evaluations

## Scenario 1: Choose A Framework

Query: "I have a new locomotion task. Should I train it with RSL-RL, RL-Games, SKRL, or SB3?"

Expected behavior:

- Asks about task type, observation space, action space, desired framework, and existing nearby examples.
- Points to official training docs and maintained agent configs.
- Recommends starting from the closest existing task config.

Known failure modes:

- Recommends hyperparameters without checking any maintained examples.
- Mixes framework-specific config formats.

## Scenario 2: Smoke Train A New Task

Query: "My environment imports now. Help me run the first training test."

Expected behavior:

- Runs or proposes reset/step validation before training.
- Uses documented Isaac Lab training entry points.
- Keeps environment count and run length small for the first training check.

Known failure modes:

- Starts a full-scale training run before checking shape and reset behavior.
- Uses a raw `python` command instead of the Isaac Lab wrapper.

## Scenario 3: Visual Observations

Query: "Train a policy using camera observations."

Expected behavior:

- Checks renderer, sensor config, observation shape, memory cost, and framework support.
- Points to sensor docs and camera-enabled examples.
- Recommends a small environment count until the visual pipeline is stable.

Known failure modes:

- Treats camera observations like low-dimensional state without checking rendering requirements.
- Ignores GPU memory and renderer constraints.

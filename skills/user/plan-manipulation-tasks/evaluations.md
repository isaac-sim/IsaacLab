# Manipulation Task Evaluations

## Scenario 1: Contact-Rich Task Planning

Query: "Help me create a grasp-and-place task for a new gripper."

Expected behavior:

- Decomposes the task into reach, align, grasp, lift, transport, and place phases.
- Requires asset physics and reset checks before training.
- Recommends controlled-frame observations and phase-specific rewards.
- Routes command details to environment and training skills.

Known failure modes:

- Proposes one monolithic reward before validating the scene and action contract.
- Ignores gripper contacts, support collision, or reset geometry.

## Scenario 2: Insertion Failure

Query: "My insertion policy reaches the hole but never inserts."

Expected behavior:

- Checks action interface, task-frame alignment, collision geometry, and controlled-frame observations.
- Separates approach/alignment from descent and insertion-depth success.
- Recommends deterministic state traces before reward-scale tuning.

Known failure modes:

- Tunes PPO first.
- Uses object-center distance as the only diagnostic.

## Scenario 3: Visual Success But Bad Metrics

Query: "The policy looks like it solved the task, but the success metric is zero."

Expected behavior:

- Compares success reward, termination, and evaluation metric geometry.
- Checks offsets, target frames, and thresholds.
- Recommends logging task-state metrics from rollout.

Known failure modes:

- Declares success from visual inspection alone.
- Changes metric thresholds without checking reward and termination definitions.

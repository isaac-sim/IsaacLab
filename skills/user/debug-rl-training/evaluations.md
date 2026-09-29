# RL Debugging Evaluations

## Scenario 1: Reward Hacking

Query: "My reward goes up, but the robot never completes the task."

Expected behavior:

- Separates reward curves from task success metrics.
- Audits per-term rewards against task phases.
- Checks success reward, termination, and metric geometry.
- Recommends deterministic rollout or state traces before retuning PPO.

Known failure modes:

- Declares the run successful from reward alone.
- Changes PPO hyperparameters before checking reward and observation semantics.

## Scenario 2: Checkpoint Mismatch

Query: "My old checkpoint no longer loads after I changed observations."

Expected behavior:

- Compares current observation/action spaces with the checkpoint training config.
- Treats shape mismatch as an environment-contract issue.
- Recommends retraining or reverting unintended contract changes.

Known failure modes:

- Patches the runner to ignore shape mismatches.
- Assumes the checkpoint is corrupt before checking environment dimensions.

## Scenario 3: One-Variable Ablation

Query: "I changed reward, observations, and reset randomization. Which run should I trust?"

Expected behavior:

- Marks the run as exploratory because multiple variables changed.
- Recommends control runs or one-variable ablations.
- Defines success metrics before launching the next run.

Known failure modes:

- Attributes behavior to one change without controls.
- Chooses checkpoints only from total reward.

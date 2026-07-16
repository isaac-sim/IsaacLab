# RL Debugging Examples

## Reward Improves But Success Does Not

Input: a run has increasing reward, but the success rate stays near zero.

Expected workflow:

1. Inspect per-term rewards and success metric definitions.
2. Check whether dense shaping saturates before task completion.
3. Compare success reward, termination, and metric geometry.
4. Run deterministic playback and inspect task-state traces.
5. Change only the smallest reward or observation issue before retraining.

## Checkpoint Shape Mismatch

Input: playback or resume fails after changing observations.

Expected workflow:

1. Print current action and observation spaces.
2. Compare them to the checkpoint's saved config.
3. Revert accidental environment-contract changes or retrain.
4. Do not patch the runner to ignore shape mismatches.

## Contact-Rich Manipulation Stall

Input: a manipulation policy reaches the object but cannot grasp, insert, or lift.

Expected workflow:

1. Check controlled-frame, object, goal, gripper, and contact observations.
2. Validate reset geometry and gripper/object initial state.
3. Inspect action scaling and controlled joints or body names.
4. Confirm contacts and physics are stable before tuning reward scales.
5. Train one subtask at a time if end-to-end behavior is too hard to interpret.

# Manipulation Task Examples

## Reaching Before Grasping

Input: a user wants to train a grasp-and-lift task from scratch.

Recommended staging:

1. Validate reachability to the pregrasp pose without object contact.
2. Add alignment rewards and controlled-frame observations.
3. Validate gripper open and close behavior with a scripted command.
4. Add contact or hold rewards only after grasp geometry is reliable.
5. Add lift or transport success last.

## Insertion Task

Input: a user wants a peg-in-hole task.

Recommended staging:

1. Separate approach, alignment, descent, contact, and insertion-depth phases.
2. Use task-frame or keypoint geometry instead of generic object-center distance.
3. Validate the action interface can move in the insertion direction.
4. Check that collision geometry permits insertion under the intended backend.
5. Select checkpoints by insertion-depth metrics and sustained success, not reward alone.

## Object Falls Through Table

Input: an object falls through a visually correct table.

Recommended staging:

1. Check whether the table is visual-only.
2. Add or reference explicit support collision geometry.
3. Validate object height and contacts immediately after reset.
4. Only resume training after the physics scene is valid.

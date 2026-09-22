# Joint Pose Diagnostics Examples

## Gripper Fingers Should Point Down

Input: "The gripper should start with the fingers pointing down."

Expected workflow:

1. Identify the finger, tool, or gripper-base body that represents the visual request.
2. Measure local body axes in world frame after reset.
3. Define the target direction, such as local `+Z` or `-Z` aligned to world `[0, 0, -1]`.
4. Sweep wrist joints inside joint limits.
5. Patch `init_state.joint_pos` only after the measured axis meets the tolerance.

## Camera Should Look At Object

Input: "The wrist camera should face the cube at reset."

Expected workflow:

1. Confirm the camera or camera mount is an exposed body or sensor frame.
2. Measure camera forward axis and object position after reset.
3. Compare the camera forward axis with the camera-to-object direction.
4. Adjust wrist or mount joints only if the camera frame is controlled through articulation joints.
5. Validate the result with a reset smoke and state measurement.

## Ambiguous Tool Axis

Input: "Make the tool point forward."

Expected workflow:

1. Ask or infer which body and local axis represents "tool forward".
2. If ambiguous, measure multiple candidate axes and explain the selected convention.
3. Report the final measured axis, dot product, and any reachability tradeoff.

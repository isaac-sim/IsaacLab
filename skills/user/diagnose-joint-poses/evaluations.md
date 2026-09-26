# Joint Pose Diagnostics Evaluations

## Scenario 1: Gripper Orientation

Query: "Set the initial gripper pose so the fingers point downward."

Expected behavior:

- Converts the request into a measured body-axis target.
- Inspects actual articulation body and joint names.
- Sweeps candidate joints within limits.
- Reports measured before/after alignment.

Known failure modes:

- Guesses joint values from a screenshot.
- Assumes USD prim names are articulation body names.

## Scenario 2: Pose Fix Breaks Reachability

Query: "The wrist now points the right way, but the robot can no longer reach the object."

Expected behavior:

- Treats axis alignment and target reachability as separate acceptance checks.
- Rechecks body-to-target distance or task-specific reach metric.
- Suggests another joint combination or reset pose instead of only maximizing axis alignment.

Known failure modes:

- Optimizes orientation while ignoring task reachability.
- Changes observations or actions unnecessarily.

## Scenario 3: Ambiguous Frame Request

Query: "Make the tool face forward."

Expected behavior:

- Identifies ambiguity in body and local-axis convention.
- Measures candidate axes and states which convention is used.
- Avoids claiming visual success without state evidence.

Known failure modes:

- Picks a body and axis silently.
- Reports only a rendered view.

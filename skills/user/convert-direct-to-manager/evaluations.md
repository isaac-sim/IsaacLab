# Direct To Manager Evaluations

## Scenario 1: Convert A Validated Direct Locomotion Task

Query: "My direct Ant task runs now. Convert it to the manager workflow so rewards and observations are reusable."

Expected behavior:

- Keeps the direct task as a comparison baseline.
- Maps scene, actions, observations, rewards, resets, terminations, and physics presets into manager configs.
- Uses the maintained Ant direct and manager pair as a reference.
- Runs random-agent and short-training smoke tests for the manager-based task.

Known failure modes:

- Deletes or rewrites the direct baseline before parity is checked.
- Changes observation ordering or scaling without calling out the policy interface change.
- Leaves reward logic buried in an environment subclass instead of MDP terms.

## Scenario 2: Convert A Direct Isaac Gym Migration

Query: "The Isaac Gym migration runs in direct form. Now make it an Isaac Lab-style task."

Expected behavior:

- Confirms direct reset, step, and short training have already passed.
- Recommends manager-based conversion because reusable managers are the main Isaac Lab task-framework benefit.
- Moves command sampling, randomization, observations, rewards, and terminations into manager terms incrementally.

Known failure modes:

- Starts manager conversion before the direct migration is a working parity baseline.
- Copies legacy Isaac Gym tensor code into manager config classes without reusable MDP functions.

## Scenario 3: Decide Not To Convert Yet

Query: "My direct environment still has shape errors, but I want it manager-based."

Expected behavior:

- Routes back to direct environment debugging first.
- Explains that manager conversion should happen after construction, reset, step, and short training pass.
- Suggests the smallest random-agent smoke test needed before conversion.

Known failure modes:

- Hides direct bugs by moving code into managers.
- Starts a larger refactor before the task has a trusted baseline.

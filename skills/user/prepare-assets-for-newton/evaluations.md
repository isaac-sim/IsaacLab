# Newton Asset Preparation Evaluations

## Scenario 1: Placeholder Inertia

Query: "My PhysX robot runs, but Newton reports placeholder inertia."

Expected behavior:

- Establishes a PhysX baseline.
- Audits authored mass, inertia, and center of mass.
- Recommends fixing authored USD physics metadata or producing a local package.
- Requires task-level Newton validation after the asset audit.

Known failure modes:

- Treats PhysX runtime success as proof of Newton readiness.
- Suppresses warnings without fixing asset metadata.

## Scenario 2: Asset Imports But Control Fails

Query: "The converted robot spawns under Newton, but the policy actions do nothing."

Expected behavior:

- Checks actuator joint patterns, controller body names, and action dimensions.
- Runs zero-action and small nonzero-action rollouts.
- Separates asset import success from control readiness.

Known failure modes:

- Keeps changing USD mass properties when the task action config is stale.
- Declares the asset ready after standalone import only.

## Scenario 3: Task-Level Failure

Query: "The object passes a standalone Newton check but fails inside my environment."

Expected behavior:

- Validates the exact task spawn path and overrides.
- Audits support collision, contact materials, and nested references.
- Checks reset and first-step finite state in the target task.

Known failure modes:

- Assumes standalone USD parsing covers task-level material and collision overrides.
- Ignores support geometry and contact-relevant scene assets.

# Backend Selection Evaluations

## Scenario 1: Isaac Gym Parity

Query: "I am porting an Isaac Gym task. Should I start with PhysX or Newton?"

Expected behavior:

- Recommends PhysX first when matching Isaac Gym behavior.
- Maps simulation parameters through Isaac Lab cfg schemas.
- Defers Newton support until the direct PhysX migration is stable unless the user explicitly targets Newton.

Known failure modes:

- Starts with Newton without explaining behavior differences.
- Copies Isaac Gym PhysX keys directly into Isaac Lab configs.

## Scenario 2: Add Presets

Query: "My task works on PhysX. Add Newton support too."

Expected behavior:

- Adds backend-specific presets or config variants.
- Checks sensors, contacts, actuators, terrain, and randomization events for backend support.
- Runs reset/step smoke tests on both backends.

Known failure modes:

- Adds runtime conditionals throughout task logic instead of config presets.
- Assumes event randomization works identically on both backends.

## Scenario 3: Renderer Or Sensor Issue

Query: "My camera observations work in one mode but fail with another backend or renderer."

Expected behavior:

- Separates physics backend, renderer, sensor type, and device assumptions.
- Points to relevant backend and sensor docs.
- Recommends a small reproducible smoke test before training.

Known failure modes:

- Treats all visual failures as physics backend bugs.
- Ignores renderer and app-launch requirements.

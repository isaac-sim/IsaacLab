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

## Scenario 4: Concrete Backend Smoke Test

Query: "Give me a command to check whether my Ant task runs on Newton before training."

Expected behavior:

- Starts by listing or checking available presets for the task.
- Uses a small `random_agent.py` rollout with a `physics=...` selector.
- Warns not to guess preset names that are not exposed by the task.
- Recommends repeating the same smoke test on PhysX for comparison.

Known failure modes:

- Starts training before reset/step validation.
- Invents backend selector names without checking task presets.
- Treats Newton and PhysX contact behavior as directly comparable without validation.

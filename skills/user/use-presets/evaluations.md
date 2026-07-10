# Preset System Evaluations

## Scenario 1: No Preset Needed

Query: "I have a simple direct task that only runs on PhysX. Should I add PresetCfg?"

Expected behavior:

- Recommends a plain config without `PresetCfg`.
- Explains that presets are useful only when there are meaningful named variants.
- Shows the simplified `SimulationCfg(physics=PhysxCfg())` pattern.

Known failure modes:

- Adds a preset wrapper with only one variant.
- Claims every task must use presets.

## Scenario 2: Add PhysX And Newton Variants

Query: "Make this locomotion environment support PhysX and Newton MJWarp."

Expected behavior:

- Uses a `PhysicsCfg(PresetCfg)` wrapper.
- Provides `default`, `physx`, and `newton_mjwarp` variants.
- Keeps solver-specific values in the preset definitions.
- Recommends random-agent smoke tests with `physics=physx` and `physics=newton_mjwarp`.

Known failure modes:

- Copies PhysX parameters directly into Newton.
- Uses runtime conditionals instead of config variants.
- Omits the default preset.

## Scenario 3: Camera Data-Type Presets

Query: "Expose RGB and depth versions of my camera task."

Expected behavior:

- Uses a domain preset selected by `presets=rgb` or `presets=depth`.
- Updates observation shape per data type.
- Calls out renderer and camera requirements.
- Points to maintained Cartpole camera preset examples.

Known failure modes:

- Uses `physics=` for a camera data-type preset.
- Omits the `renderer=` selector for a camera task.
- Leaves observation shape unchanged when switching RGB to depth.

## Scenario 4: Discover Available Options

Query: "What preset names can I pass for this task?"

Expected behavior:

- Recommends listing available presets before guessing names.
- Explains `physics=`, `renderer=`, and `presets=` selector categories.
- Warns that not every task supports every common preset name.

Known failure modes:

- Invents preset names not exposed by the task.
- Confuses typed selectors with task-specific domain presets.

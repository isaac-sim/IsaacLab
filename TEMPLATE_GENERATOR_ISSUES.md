# Template Generator Issue Record

| Issue | Resolution |
| --- | --- |
| Ctrl+C prints a traceback from the zero and random agents. | Catch `KeyboardInterrupt`, close the environment, and exit cleanly. |
| Generated Cartpole configs use deprecated rigid-body and articulation schema classes. | Current templates inherit the corrected shared Cartpole configuration; keep a regression check for deprecated names. |
| Environment listing requires a project-local script. | Add `isaaclab list_envs` with project filtering and preset support; retain the scripts as wrappers. |
| [PR #3698](https://github.com/isaac-sim/IsaacLab/pull/3698) adds project-local simple-agent instructions. | Closed as superseded because the package CLI now ships and documents both agents. |
| Generated metadata uses the Isaac Lab developers as authors. | Prompt for authors and write them to package, extension, and license metadata. |
| The generated license attribution is hard-coded. | Keep Isaac Lab's BSD-3-Clause license and render the selected authors and current year. |
| The task entry-point key is quoted unnecessarily. | Render the validated identifier as a bare TOML key. |
| Permanent source files describe themselves as generated artifacts. | Replace that wording with each module's purpose. |
| Every project contains the Cartpole example. | Add Cartpole and Blank initial-content choices; Blank emits only project infrastructure. |
| [PR #2315](https://github.com/isaac-sim/IsaacLab/pull/2315) proposes a side-effectful custom-asset flow. | Add a packaged `assets/data` directory and asset-path constant without downloads or simulator launches. |
| Generator tests activate unrelated suites under `tools/`. | Move focused tests to the Isaac Lab CLI suite; they run in well under one second. |
| [Issue #3223](https://github.com/isaac-sim/IsaacLab/issues/3223) requests automation. | Add opt-in `--non_interactive` arguments while keeping prompts as the default. |
| Generated projects omit extras such as `rerun`, `viser`, and `all`. | Forward every extra exposed by the active Isaac Lab source tree or installed distribution. |

The reported Newton shape-color and Cartpole mass/inertia warnings also occur with Isaac Lab's built-in Cartpole task. They require shared backend or asset fixes and are not suppressed by the generator.

# Template Generator Issue Record

This file tracks reported template-generator issues, their resolution, and the validation used to confirm each fix.
Update it as new issues are found or existing fixes change.

## Resolved issues

### Random and zero agents print a traceback on Ctrl+C

- **Reported behavior:** Stopping `isaaclab random_agent` with Ctrl+C propagated `KeyboardInterrupt` and printed a traceback.
- **Fix:** Catch `KeyboardInterrupt` around the agent loop, print a short stopped message, and close the environment in a `finally` block.
- **Files:**
  - `source/isaaclab_rl/isaaclab_rl/entrypoints/simple_agents.py`
  - `source/isaaclab_rl/test/test_entrypoints.py`
- **Validation:** A live `uv run isaaclab random_agent --task Isaac-Cartpole --num_envs 1 --viz none` session exited with status 0 after Ctrl+C and printed no traceback. The focused entry-point suite passes 47 tests.

### Generated projects use deprecated simulation schema classes

- **Reported behavior:** `pytest` emitted deprecation warnings for `RigidBodyPropertiesCfg` and `ArticulationRootPropertiesCfg` from generated task configuration.
- **Fix:** Generate the current solver-common base schema configuration classes instead of the deprecated classes.
- **Files:** Task templates under `tools/template/templates/tasks/`.
- **Validation:** A freshly generated Cartpole project passes its registration test without either schema deprecation warning. Generator tests also reject either deprecated class in generated output.

### Environment listing requires a project-local script

- **Reported behavior:** Users had to run `uv run python scripts/list_envs.py --show_presets`; the functionality was absent from the Isaac Lab CLI.
- **Fix:** Add `uv run isaaclab list_envs`, with `--show_presets`, `--keyword`, and `--all`. When run inside a downstream project, it reads the nearest `pyproject.toml` and shows tasks registered by that project's `isaaclab.tasks` entry point by default. Existing scripts are compatibility wrappers.
- **Files:**
  - `source/isaaclab/isaaclab/cli/commands/list_envs.py`
  - `source/isaaclab/isaaclab/cli/__init__.py`
  - `scripts/environments/list_envs.py`
  - `tools/template/templates/external/list_envs.py`
- **Validation:** The focused CLI suites pass 17 tests. The command also exits successfully when no tasks match a keyword.

### Dummy-agent documentation PR targets obsolete project-local scripts

- **Reported context:** [PR #3698](https://github.com/isaac-sim/IsaacLab/pull/3698) adds external-project instructions for `scripts/zero_agent.py` and `scripts/random_agent.py`, and corrects internal paths to `scripts/environments/`.
- **Finding:** Isaac Lab now ships `zero_agent` and `random_agent` through its package CLI. Current generated projects and current documentation use `uv run isaaclab zero_agent` and `uv run isaaclab random_agent`. The PR's only changed page, `docs/source/overview/own-project/template.rst`, has also been removed and replaced by `docs/source/developer-tools/template_generator.rst`.
- **Resolution:** PR #3698 was closed as superseded, with a comment explaining the replacement workflow. Its desired user workflow is already available through the package CLI and documented for external projects without copying or locating project-local agent scripts.

### Generated projects do not collect author information

- **Reported behavior:** Project metadata always named the Isaac Lab Project Developers as the author.
- **Fix:** Prompt for one or more comma-separated authors and render them into `pyproject.toml`, `LICENSE`, and extension metadata. TOML-sensitive characters are escaped safely.
- **Files:**
  - `tools/template/cli.py`
  - `tools/template/generator.py`
  - `tools/template/templates/external/pyproject.toml.jinja`
  - `tools/template/templates/external/LICENSE`
  - `tools/template/templates/extension/config/extension.toml`
- **Validation:** Generator tests parse the resulting TOML and cover multiple authors and a quoted author name.

### Generated project license attribution is hard-coded

- **Reported behavior:** The generated license used Isaac Lab's copyright attribution rather than the new project's authors.
- **Resolution:** Keep the same BSD-3-Clause license used by Isaac Lab, as clarified in the request, and render the current year and selected project authors into its copyright line.
- **Files:** `tools/template/templates/external/LICENSE` and `tools/template/generator.py`.
- **Validation:** Generator tests assert the selected authors appear in the generated license.

### Task entry-point keys are unnecessarily quoted

- **Reported behavior:** Generated TOML used `"project_name" = "project_name.tasks"` for an identifier-safe project name.
- **Fix:** Render the validated project identifier as an unquoted TOML key.
- **File:** `tools/template/templates/external/pyproject.toml.jinja`.
- **Validation:** Generator tests parse the entry-point table and assert the expected mapping.

### Templates describe project files as generated artifacts

- **Reported behavior:** Docstrings repeatedly used phrases such as "generated project" and "generated task" in permanent project source files.
- **Fix:** Replace those phrases with descriptions of each module's actual purpose.
- **Files:** Templates under `tools/template/templates/`.
- **Validation:** A case-insensitive search of the template tree finds no remaining use of `generated`.

### Every new project contains the Cartpole example

- **Reported behavior:** The generator could not create an empty project structure without Cartpole task, MDP, agent, and configuration files.
- **Fix:** Add an `Initial project content` prompt with `Cartpole` and `Blank` choices. `Cartpole` remains the default runnable example. `Blank` creates packaging, task discovery, tests, and development tooling, skips task/workflow/RL prompts, and emits no example task family.
- **Files:**
  - `tools/template/cli.py`
  - `tools/template/generator.py`
  - `tools/template/templates/external/README.md`
  - `tools/template/templates/external/test_registration`
  - `tools/template/templates/external/__init__package`
  - `tools/template/templates/external/__init__tasks`
- **Validation:** Fresh Cartpole and Blank projects both pass their registration tests. The blank package contains no Cartpole task files.

### Custom assets have no clear home in external projects

- **Reported context:** [PR #2315](https://github.com/isaac-sim/IsaacLab/pull/2315) proposes a custom-USD prompt, downloads copies of Cartpole assets, launches Isaac Sim during generation, patches copied asset configurations, and rewrites generated task imports.
- **Finding:** The underlying need remains valid, but the implementation targets the removed `source/` project layout and conflicts with the current generator. It also treats copied Cartpole files as custom assets, introduces network and simulator side effects, can leave invalid placeholder USD files after download failures, and always imports both copied robot modules even when only one was generated.
- **Fix extracted from the PR:** Every external project now includes `src/<project>/assets`, a package-relative `<PROJECT>_ASSETS_DIR` constant pointing to its `data` directory, and README instructions for project-owned assets. This works for both Blank and Cartpole projects without another prompt, download, simulator launch, or task rewrite. Generated `.gitattributes` already routes USD formats through Git LFS.
- **Validation:** The focused generator suite includes assertions for this layout. A sample USDA file placed in a generated Blank project's asset directory was present in the wheel produced by `uv build --wheel`.
- **PR status:** The feature request is still valid; PR #2315 is not suitable to merge as written. It can be closed as superseded after this replacement is accepted.

### Template generator tests are slow or invoke unrelated repository tests

- **Reported behavior:** Collecting the old test under `tools/` activated `tools/conftest.py`, which launches the repository test orchestrator and unrelated suites.
- **Fix:** Move the focused generator tests into the Isaac Lab CLI test package and reduce them to observable prompt and generated-output behavior.
- **Files:**
  - Removed `tools/template/test_cli.py`
  - Added `source/isaaclab/test/cli/test_template_generator.py`
- **Validation:** `uv run python -m pytest source/isaaclab/test/cli/test_template_generator.py -q` passes 15 tests in approximately 0.13 seconds.

### Template generation cannot be automated without answering prompts

- **Reported context:** [Issue #3223](https://github.com/isaac-sim/IsaacLab/issues/3223) requests a non-interactive flow with arguments such as project path, workflow, and RL algorithm.
- **Fix:** Add an opt-in `--non-interactive` mode. Interactive prompting remains the default, and content arguments are rejected without the flag. External automation requires project path, name, and at least one author. Cartpole mode defaults to manager-based single-agent, RSL-RL, and PPO while accepting repeatable workflow, library, and algorithm overrides. Blank mode creates only the project structure and rejects task-specific options. Internal task automation is available from source checkouts.
- **Files:**
  - `tools/template/cli.py`
  - `source/isaaclab/test/cli/test_template_generator.py`
  - `docs/source/developer-tools/template_generator.rst`
- **Validation:** The focused suite covers the opt-in guard, default Cartpole specification, Blank specification, and explicit direct multi-agent SKRL selection. The CLI's `--help` output lists every automation argument.

### Generated projects omit Isaac Lab optional extras

- **Reported behavior:** A project generated from Isaac Lab 3.0 only declared `isaacsim`, `ov`, `ovphysx`, and `ovrtx`. Commands such as `uv run --extra isaacsim --extra rerun ...` failed before launch because the generated project did not define `rerun`; `viser` and `all` were also absent.
- **Fix:** Populate the generated `[project.optional-dependencies]` table from the complete extra set exposed by the active Isaac Lab package. Source generation reads the checkout's root `pyproject.toml`; installed-wheel generation reads `Provides-Extra` from distribution metadata. Each project extra forwards to the same version/source dependency, for example `rerun = ["isaaclab[rerun]==<version>"]`.
- **Files:**
  - `tools/template/generator.py`
  - `tools/template/templates/external/pyproject.toml.jinja`
  - `tools/template/templates/external/README.md`
  - `docs/source/developer-tools/template_generator.rst`
  - `source/isaaclab/test/cli/test_template_generator.py`
- **Validation:** A source-generated project contains all 22 extras declared by the current checkout, including `all`, `rerun`, and `viser`. `uv run --extra rerun --extra viser --extra all --no-sync ...` accepts all three from that project. Focused tests assert the complete source set, metadata discovery, and the version-pinned installed-package mapping.

## Investigated issues requiring a shared asset or backend fix

### Newton shape-color replacement FutureWarning

- **Reported behavior:** Newton emits a `FutureWarning` from `replace_newton_builder_shape_colors` while cloning environments.
- **Finding:** The same warning occurs with Isaac Lab's built-in Cartpole task and is deliberately emitted by the Newton compatibility path. Existing Newton tests assert this warning.
- **Current status:** No generator-specific change. Suppressing it in generated projects would hide a shared backend diagnostic without removing its cause.
- **Likely fix location:** The Newton importer/cloner compatibility implementation when shape-color replacement is no longer needed.

### Newton Cartpole mass and inertia warnings

- **Reported behavior:** Newton warns about the Cartpole slider's negative/invalid mass and approximated inertia for the cart and pole.
- **Finding:** The warnings reproduce with Isaac Lab's built-in Cartpole asset. The shared USD asset relies on inferred mass/inertia and instanceable collider references that Newton currently diagnoses differently.
- **Current status:** No generator-specific change. Adding arbitrary mass or inertia values to the template would fork the shared asset's physical behavior and could change training results.
- **Likely fix location:** The shared Cartpole asset or Newton USD mass/inertia import behavior, followed by physics and policy regression validation.

## Repository validation status

- `git diff --check`, Ruff, Ruff Format, TOML validation, and codespell pass for the changed files.
- `uv run isaaclab -f` passes changed-code checks but reports pre-existing changelog divergence and missing fragments for unrelated packages on this branch. Those unrelated files were not changed as part of this work.

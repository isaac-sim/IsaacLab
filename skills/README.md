# Isaac Lab Agent Skills

Isaac Lab skills are repo-owned instructions that help agents follow project workflows and user-facing Isaac Lab patterns. They are guidance assets, not runtime Python packages.

## Catalog

Developer skills:

- `developer/pr-workflow/`: prepare changes for review using Isaac Lab's PR, commit, changelog, and validation conventions.
- `developer/changelog-fragments/`: add and validate package changelog fragments.
- `developer/coding-style/`: apply Isaac Lab coding style, API design, docstring, type-hint, lazy export, and contribution conventions.

User skills:

- `user/install-isaac-lab/`: install Isaac Lab following the current install docs — automatic uv setup, downloaded Isaac Sim package, source build, Isaac Lab wheel, legacy isaaclab.sh installer, or Docker — across Linux (x86_64, aarch64) and Windows 11.
- `user/migrate-from-isaac-gym/`: migrate Isaac Gym tasks, assets, and training workflows to Isaac Lab.
- `user/migrate-2x-to-3x/`: migrate Isaac Lab 2.x projects to Isaac Lab 3.0 using the official migration guide.
- `user/domain-randomization-events/`: implement domain randomization through Isaac Lab event terms.
- `user/create-environments/`: create manager-based Isaac Lab environments by default, with direct environments for special cases.
- `user/convert-direct-to-manager/`: convert validated direct Isaac Lab environments into manager-based task configurations.
- `user/train-rl-agents/`: configure and run Isaac Lab reinforcement learning workflows.
- `user/debug-rl-training/`: diagnose RL rewards, task metrics, checkpoint compatibility, and training experiments.
- `user/plan-manipulation-tasks/`: stage manipulation tasks through scene, reset, action, reward, and behavior gates.
- `user/use-sensors-actuators/`: add sensors, sensor observations, and actuator models to tasks.
- `user/diagnose-joint-poses/`: measure and correct robot initial joint poses from semantic or visual pose requests.
- `user/select-backends/`: choose and validate PhysX, Newton, and backend-specific task presets.
- `user/use-presets/`: define and use preset configurations for multi-backend and variant-rich tasks.
- `user/prepare-assets-for-newton/`: validate and prepare PhysX-compatible USD assets for Newton task workflows.
- `user/setup-troubleshooting/`: route installation, verification, and setup issues to official docs and canonical commands.

Planned user skills:

- `user/import-robot-urdf-mjcf/`

## Discovery

Codex and Claude discover these skills automatically from project-native aliases:

- Codex scans `.agents/skills/<name>/`.
- Claude scans `.claude/skills/<name>/`, which links to the same alias set.

The aliases are named from each `SKILL.md` frontmatter `name` and point back to the canonical skill directory under `skills/`. This keeps one maintained copy of each skill and preserves repository-relative references. Do not flatten-copy skills into an agent's global skill directory.

Agents that do not support native skill discovery should start at this file. Match the user's request against each `SKILL.md` frontmatter `description`, then read only the selected skill and its directly linked files. When one skill routes to another, use the frontmatter `name` as the stable identifier and the catalog path as the file location.

## Common Import Paths

Use these current import paths before searching for alternatives:

| Concept | Import path |
| --- | --- |
| Direct RL environment config | `from isaaclab.envs import DirectRLEnvCfg` |
| Direct multi-agent environment config | `from isaaclab.envs import DirectMARLEnvCfg` |
| Manager-based RL environment config | `from isaaclab.envs import ManagerBasedRLEnvCfg` |
| Event term config | `from isaaclab.managers import EventTermCfg as EventTerm` |
| Scene entity config | `from isaaclab.managers import SceneEntityCfg` |
| Preset config | `from isaaclab_tasks.utils import PresetCfg` |
| Simulation config | `from isaaclab.sim import SimulationCfg` |
| PhysX physics config | `from isaaclab_physx.physics import PhysxCfg` |
| Newton physics config | `from isaaclab_newton.physics import NewtonCfg` |
| Base contact sensor config | `from isaaclab.sensors import ContactSensorCfg` |
| PhysX contact sensor config | `from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg` |
| Newton contact sensor config | `from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg` |
| Ray caster config | `from isaaclab.sensors import RayCasterCfg` |
| Tiled camera config | `from isaaclab.sensors import TiledCameraCfg` |
| Implicit actuator config | `from isaaclab.actuators import ImplicitActuatorCfg` |
| Core schema fragments and base cfgs | `from isaaclab.sim import schemas` |
| PhysX schema cfgs | `from isaaclab_physx.sim import schemas as physx_schemas` |
| Newton schema cfgs | `from isaaclab_newton.sim import schemas as newton_schemas` |

## Authoring Rules

Every skill directory must contain a `SKILL.md` file with frontmatter:

```yaml
name: isaaclab-example-skill
description: Does a specific Isaac Lab task. Use when the user mentions the task or related trigger terms.
audience: user
status: stable
owners:
  - isaaclab-maintainers
```

Directory slugs are stable canonical file paths for humans and reviewers. The frontmatter `name` is the agent discovery identifier, must match the aliases under `.agents/skills/`, and should be used when one skill routes to another. `.claude/skills` exposes the same aliases to Claude.

Required `SKILL.md` sections:

- `When To Use`
- `Workflow`
- `Validation`
- `Maintenance`
- `References`

User-facing skills must also link to `evaluations.md` with at least three representative scenarios. Each scenario must include a sample query, expected behavior, and known failure modes or pass/fail criteria.

Keep skills concise. Use `SKILL.md` for the main workflow and link directly to one-level files such as `reference.md`, `examples.md`, or `evaluations.md` for details. Use forward-slash paths, avoid time-sensitive wording, and provide one recommended default before listing alternatives.

Keep skills synchronized by making official docs and source code the source of truth. If a skill needs documentation-level details, update `docs/source/` or a maintained source example first, then link to it from the skill. The `Maintenance` section must name the authoritative files that should be reviewed when code changes. When adding, removing, or renaming a skill, update its `.agents/skills/<name>` alias; the validator checks both Codex and Claude discovery paths.

Skills should add agent-specific routing, sequencing, validation checks, and decision points. They should not vendor installation guides, API catalogs, generated logs, hardware benchmark reports, or large tutorial copies.

Run the validator before submitting skill changes:

```bash
uv run --no-project python tools/skills/cli.py check
```

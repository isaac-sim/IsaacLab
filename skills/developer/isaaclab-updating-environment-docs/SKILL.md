---
name: isaaclab-updating-environment-docs
description: Keeps the illustrated environment catalog and generated comprehensive table synchronized with task registrations and preset selectors. Use when adding or renaming an Isaac Lab environment, changing its physics, renderer, or domain presets, or reviewing environment documentation completeness.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Updating Environment Documentation

## When To Use

Use this skill whenever a registered environment is added or renamed, or when its accepted ``physics=``, ``renderer=``, or ``presets=`` selectors change.

## Workflow

1. Read the task registration and configuration to identify its training ID, inference ID, workflow, RL-library entry points, and selectable presets. Use versionless ``Isaac-`` IDs for core tasks and versionless ``IsaacContrib-`` IDs for contributed tasks. Preserve released IDs as deprecated aliases when renaming is necessary. Distinguish selectable presets from fixed configuration: if a task supports only one hard-wired backend, leave its Presets cell empty instead of adding or documenting a single-option selector.
2. Update the appropriate illustrated category table in ``docs/source/overview/environments.rst``. Add a concise description, exact selector groups, source-link substitutions, and a representative screenshot under ``docs/source/_static/tasks/`` when the world is new. Capture catalog screenshots with the Kit visualizer so their rendering matches the existing gallery unless the section deliberately documents another visualizer.
3. Run the registry-backed updater:

   ```bash
   uv run python tools/update_environments_rst.py
   ```

   This rewrites the comprehensive table from the Gym registry and groups names under ``physics=``, ``renderer=``, and ``presets=``.
4. Review the generated diff. Do not edit content between the comprehensive-list marker comments by hand.
5. Confirm the illustrated table and comprehensive table both contain the task and agree with the task's accepted selectors.
6. When the documentation update is part of a PR, build the docs and require a clean result. The docs target runs Sphinx with warnings treated as errors, so any warning or error must be resolved before publishing the PR.

## Validation

Run:

```bash
uv run python tools/update_environments_rst.py --check
uv run python scripts/environments/list_envs.py --show_presets
uv run isaaclab -d
uv run --no-project python tools/skills/cli.py check
```

Require ``uv run isaaclab -d`` to exit successfully with no warnings or errors. Inspect the compiled
environment catalog page in the documentation build output and verify that every new image and
substitution target renders correctly.

## Maintenance

Keep this workflow synchronized with the environment catalog markers, the registry collector, the preset CLI, and the environment-list script. When those tools change, update the source documentation behavior first and keep this skill as a concise routing checklist.

## References

- [Environment catalog](../../../docs/source/overview/environments.rst)
- [Environment documentation collector](../../../tools/environ_docs.py)
- [Environment documentation updater](../../../tools/update_environments_rst.py)
- [Environment list script](../../../scripts/environments/list_envs.py)
- [Preset CLI](../../../source/isaaclab_tasks/isaaclab_tasks/utils/preset_cli.py)

---
name: isaaclab-updating-environment-docs
description: Keeps the generated environment browser synchronized with task registrations and preset selectors. Use when adding or renaming an Isaac Lab environment, changing its physics, renderer, or domain presets, or reviewing environment documentation completeness.
license: BSD-3-Clause
metadata:
  author: Isaac Lab Team <Isaac-Lab@exchange.nvidia.com>
---

# Updating Environment Documentation

## When To Use

Use this skill whenever a registered environment is added or renamed, or when its accepted ``physics=``, ``renderer=``, or ``presets=`` selectors change.

## Workflow

1. Read the task registration and configuration to identify its training ID, inference ID, workflow, RL-library entry points, and selectable presets. Use versionless ``Isaac-`` IDs for core tasks and versionless ``IsaacContrib-`` IDs for contributed tasks. Preserve released IDs as deprecated aliases when renaming is necessary. Distinguish selectable presets from fixed configuration: if a task supports only one hard-wired backend, leave its Presets cell empty instead of adding or documenting a single-option selector.
2. Add a representative screenshot under ``docs/source/_static/tasks/`` when the world is new. Assign the image to the task's generated row in ``docs/source/_static/css/environment-browser.js`` before running the updater. Capture screenshots with the Kit visualizer so their rendering matches the existing previews unless the task deliberately documents another visualizer.
3. Run the registry-backed updater:

   ```bash
   uv run python tools/update_environments_rst.py
   ```

   This rewrites the environment-browser task rows from the Gym registry and groups names under ``physics=``, ``renderer=``, and ``presets=``. Existing preview-image assignments are preserved.
4. Review the generated diff. Do not edit selector data between the environment-browser marker comments by hand.
5. Confirm the environment browser contains the task and agrees with the task's accepted selectors.
6. When the documentation update is part of a PR, build the docs and require a clean result. The docs target runs Sphinx with warnings treated as errors, so any warning or error must be resolved before publishing the PR.

## Validation

Run:

```bash
uv run python tools/update_environments_rst.py --check
uv run python scripts/environments/list_envs.py --show_presets
uv run --isolated --extra test -- make -C docs current-docs
uv run --no-project python tools/skills/cli.py check
```

Require the combined test and documentation build to exit successfully with no warnings or errors. Inspect the
compiled environment browser in the documentation build output and verify that every new image
and task selector renders correctly.

## Maintenance

Keep this workflow synchronized with the environment-browser markers, the registry collector, the preset CLI, and the environment-list script. When those tools change, update the source documentation behavior first and keep this skill as a concise routing checklist.

## References

- [Environment browser](../../../docs/source/setup/environments.rst)
- [Environment documentation collector](../../../tools/environ_docs.py)
- [Environment documentation updater](../../../tools/update_environments_rst.py)
- [Environment list script](../../../scripts/environments/list_envs.py)
- [Preset CLI](../../../source/isaaclab_tasks/isaaclab_tasks/utils/preset_cli.py)

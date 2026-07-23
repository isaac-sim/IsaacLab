---
name: isaaclab-migrating-2x-to-3x
description: Migrates Isaac Lab 2.x projects to Isaac Lab 3.0 by routing agents through the official migration guide, current source APIs, and focused compatibility checks. Use when users mention Isaac Lab 3.0 migration, 2.x projects, quaternion order changes, ProxyArray data access, backend migration, Isaac Sim extension imports, or visualization CLI changes.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Migrating 2.x To 3.x

## When To Use

Use this skill when migrating a downstream Isaac Lab 2.x project to Isaac Lab 3.0 or diagnosing migration errors after an upgrade.

Do not copy migration tables into answers from memory. Read the official migration guide first, then inspect the current source or examples for the specific API involved.

## Workflow

1. Read the official migration guide in `docs/source/migration/migrating_to_isaaclab_3-0.rst`.
2. Identify which migration area applies: visualization CLI, backend packages, schema cfgs, quaternion order, `ProxyArray`, asset views, RSL-RL config, Isaac Sim extension enablement, or project-specific scripts.
3. Search the downstream project for old API symbols before editing.
4. For user code that imports Isaac Sim extension modules directly:
   - Prefer an Isaac Lab in-tree API when the migration guide lists one.
   - If the direct Isaac Sim import is still needed, add `from isaaclab.sim.utils import enable_extension` and call `enable_extension("<extension_name>")` after the Kit application starts and before the first import from that extension.
   - Do not import `enable_extension` from `isaacsim.core.experimental.utils.app`; that module may not be importable until its extension is already enabled.
   - For custom `.kit` files, either list every direct Isaac Sim extension dependency explicitly or enable each extension before importing its Python module.
   - Do not add extension enables to plain Python modules that can be imported before Kit starts unless the code is guarded so it runs only inside a launched Kit app.
5. Apply the smallest focused migration change.
6. Run a targeted smoke test or import test. For direct Isaac Sim imports, use an app-launched command such as `./isaaclab.sh -p -m pytest PATH_TO_TEST` or a script that starts `AppLauncher` before enabling extensions.
7. If the official docs are missing a recurring migration issue, update `docs/source/migration/migrating_to_isaaclab_3-0.rst` instead of expanding this skill with standalone documentation.

## Validation

Use this feedback loop:

```bash
uv run --with pytest python -m pytest PATH_TO_DOWNSTREAM_TEST
```

For quaternion migrations, use the repository quaternion tooling documented in the official migration guide.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/migration/migrating_to_isaaclab_3-0.rst`, `docs/source/setup/installation/index.rst`, `source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py`, and the uv-based `train` and `play` entry points. If code changes invalidate migration guidance, update the official migration document first and keep this skill as a router plus checklist.

## References

- [Reference](reference.md)
- [Supplemental checks](supplemental-checks.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Official migration guide](../../../docs/source/migration/migrating_to_isaaclab_3-0.rst)
- [RSL-RL compatibility helper](../../../source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py)

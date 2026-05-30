# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal helpers for generating environment documentation from the Gym registry.

These utilities collect RL-library entry points, preset selectors, workflow
types, and inference-task mappings for each registered Isaac Lab task. They
are used by :mod:`tools.update_environments_rst` to keep
``docs/source/overview/environments.rst`` in sync with the codebase.
"""

from __future__ import annotations

import collections
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym

from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
from isaaclab_tasks.utils.preset_target import PresetTarget

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec

# Physics-backend preset names that may be mirrored on non-physics ``PresetCfg``
# fields (for example contact-sensor presets). Those mirrors are resolved together
# with ``physics=`` when the env's physics preset declares the name, but should
# not appear under ``presets=`` when physics does not expose them.
_PHYSICS_BACKEND_MIRROR_NAMES = frozenset(
    {
        "newton_kamino",
        "newton_mjwarp",
        "newton_mjwarp_vbd",
        "newton_mjwarp_vdb",
        "ovphysx",
        "physx",
        *PresetTarget.all_legacy_aliases().keys(),
        *PresetTarget.all_legacy_aliases().values(),
    }
)

# RL libraries listed in a stable order across generated docs.
_RL_LIBRARY_ORDER = ("rl_games", "rsl_rl", "skrl", "sb3", "rlinf")

# Gym IDs reserved for inference / evaluation variants and excluded from the training list.
_INFERENCE_TASK_SUFFIXES = ("-Play-v0", "-Eval-v0")

# RL libraries not discoverable from Gym ``kwargs`` (e.g. RLinf YAML-based workflows).
RL_LIBRARY_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "Isaac-Assemble-Trocar-G129-Dex3-v0": {"rlinf": ["PPO"]},
}

# Marker comments that delimit the auto-generated section in environments.rst.
COMPREHENSIVE_LIST_START_MARKER = ".. START-AUTO-GENERATED: comprehensive-environment-list"
COMPREHENSIVE_LIST_END_MARKER = ".. END-AUTO-GENERATED: comprehensive-environment-list"


@dataclass(frozen=True)
class EnvironmentDocRow:
    """One row of the comprehensive environment list in ``environments.rst``."""

    task_name: str
    inference_task_name: str | None
    workflow: str
    rl_libraries: str
    presets: str


def is_training_task(task_id: str) -> bool:
    """Return ``True`` when *task_id* is a training (non-inference) Isaac task."""
    if "Isaac" not in task_id:
        return False
    if any(task_id.endswith(suffix) for suffix in _INFERENCE_TASK_SUFFIXES):
        return False
    if "-Benchmark-" in task_id:
        return False
    return True


def parse_rl_libraries_from_kwargs(kwargs: dict) -> dict[str, list[str]]:
    """Parse RL-library and algorithm labels from Gym registry kwargs.

    Args:
        kwargs: ``kwargs`` passed to :func:`gymnasium.register` for a task.

    Returns:
        Mapping of RL-library name to sorted algorithm labels (e.g. ``{"skrl": ["IPPO", "PPO"]}``).
    """
    agents: dict[str, set[str]] = collections.defaultdict(set)
    for key, value in kwargs.items():
        if not key.endswith("_cfg_entry_point") or key == "env_cfg_entry_point":
            continue

        stem = key[: -len("_cfg_entry_point")]
        library = None
        algo_suffix = ""
        for candidate in _RL_LIBRARY_ORDER:
            if stem == candidate:
                library = candidate
                break
            prefix = f"{candidate}_"
            if stem.startswith(prefix):
                library = candidate
                algo_suffix = stem[len(prefix) :]
                break
        if library is None:
            continue
        if algo_suffix == "with_symmetry":
            continue
        agents[library].add(_infer_algorithm(algo_suffix, value, library))

    return {library: sorted(algorithms, key=_algo_sort_key) for library, algorithms in agents.items()}


def apply_rl_library_overrides(task_id: str, agents: dict[str, list[str]]) -> dict[str, list[str]]:
    """Merge manual RL-library overrides for tasks without cfg entry points."""
    override = RL_LIBRARY_OVERRIDES.get(task_id)
    if override is None:
        return agents

    merged = dict(agents)
    for library, algos in override.items():
        combined = set(merged.get(library, [])) | set(algos)
        merged[library] = sorted(combined, key=_algo_sort_key)
    return merged


def format_rl_libraries(agents: dict[str, list[str]]) -> str:
    """Format RL libraries for an RST table cell."""
    if not agents:
        return ""
    parts = []
    for library in _RL_LIBRARY_ORDER:
        if library not in agents:
            continue
        algos = ", ".join(agents[library])
        parts.append(f"**{library}** ({algos})")
    return ", ".join(parts)


def _infer_implicit_physics_names(task_name: str) -> set[str]:
    """Return physics preset names implied by ``PresetCfg.default`` fields.

    Many env cfgs set ``default = PhysxCfg(...)`` without an explicit ``physx``
    alias. The preset CLI excludes ``default`` from help listings, but users
    still select PhysX via ``physics=physx`` (or by falling back to default).
    """
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab.physics import PhysicsCfg

    from isaaclab_tasks.utils.hydra import collect_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    names: set[str] = set()
    for fields in collect_presets(env_cfg).values():
        default = fields.get("default")
        if default is None or not isinstance(default, PhysxCfg):
            continue
        has_physics_variant = any(name != "default" and isinstance(value, PhysicsCfg) for name, value in fields.items())
        if has_physics_variant:
            names.add("physx")
    return names


def _physics_names_for_docs(task_name: str, preset_map: dict[PresetTarget, list[str]] | None) -> list[str]:
    """Return sorted physics preset names for documentation, with implicit PhysX."""
    if preset_map is None:
        return []
    names = set(preset_map.get(PresetTarget.PHYSICS, []))
    with contextlib.suppress(Exception):
        names |= _infer_implicit_physics_names(task_name)
    return sorted(names)


def _domain_presets_for_docs(preset_map: dict[PresetTarget, list[str]]) -> list[str]:
    """Return domain preset names that are not already covered by typed selectors.

    Some tasks mirror physics-backend names on non-physics ``PresetCfg`` fields
    (for example observation presets named ``physx`` / ``newton_mjwarp``, or
    contact-sensor presets named ``ovphysx``). Names that also exist on the
    env's physics preset are selected via ``physics=`` and are filtered here.
    Backend names that appear only on non-physics fields are omitted entirely
    because they are not meaningful ``presets=`` selectors for users.
    """
    physics_names = set(preset_map.get(PresetTarget.PHYSICS, []))
    typed_names = physics_names | set(preset_map.get(PresetTarget.RENDERER, []))
    domain_names: list[str] = []
    for name in preset_map.get(PresetTarget.DOMAIN, []):
        if name in typed_names:
            continue
        if name in _PHYSICS_BACKEND_MIRROR_NAMES and name not in physics_names:
            continue
        domain_names.append(name)
    return domain_names


def format_presets_rst(preset_map: dict[PresetTarget, list[str]] | None) -> str:
    """Format preset selectors for an RST ``list-table`` cell."""
    if preset_map is None:
        return ""

    groups: list[tuple[str, list[str]]] = []
    physics_names = preset_map.get(PresetTarget.PHYSICS, [])
    renderer_names = preset_map.get(PresetTarget.RENDERER, [])
    domain_names = _domain_presets_for_docs(preset_map)

    if physics_names:
        groups.append(("physics=", physics_names))
    if renderer_names:
        groups.append(("renderer=", renderer_names))
    if domain_names:
        groups.append(("presets=", domain_names))

    if not groups:
        return ""

    formatted_groups = [f"**{label}** {', '.join(f'``{name}``' for name in names)}" for label, names in groups]
    if len(formatted_groups) == 1:
        return formatted_groups[0]
    return "\n        ".join(
        f"| {group}" if index == 0 else f"  | {group}" for index, group in enumerate(formatted_groups)
    )


def get_workflow(entry_point: str) -> str:
    """Return the human-readable workflow label for a Gym entry point."""
    if "ManagerBasedRLEnv" in entry_point:
        return "Manager Based"
    return "Direct"


def find_inference_task_name(task_id: str, registry_ids: set[str]) -> str | None:
    """Return the inference/play task ID paired with *task_id*, if registered."""
    base = task_id.rsplit("-", 1)[0]
    for suffix in _INFERENCE_TASK_SUFFIXES:
        candidate = f"{base}{suffix}"
        if candidate in registry_ids:
            return candidate
    return None


def collect_environment_doc_rows(
    specs: list[EnvSpec] | None = None,
) -> list[EnvironmentDocRow]:
    """Collect documentation rows for every training Isaac task in the registry.

    Args:
        specs: Optional list of Gym specs. When ``None``, all registered specs are scanned.

    Returns:
        Sorted list of :class:`EnvironmentDocRow` entries.
    """
    if specs is None:
        specs = list(gym.registry.values())

    registry_ids = {spec.id for spec in specs}
    rows: list[EnvironmentDocRow] = []

    for spec in specs:
        if not is_training_task(spec.id):
            continue

        preset_map = enumerate_task_presets(spec.id)
        if preset_map is not None:
            preset_map = dict(preset_map)
            preset_map[PresetTarget.PHYSICS] = _physics_names_for_docs(spec.id, preset_map)
        agents = apply_rl_library_overrides(spec.id, parse_rl_libraries_from_kwargs(spec.kwargs))

        rows.append(
            EnvironmentDocRow(
                task_name=spec.id,
                inference_task_name=find_inference_task_name(spec.id, registry_ids),
                workflow=get_workflow(spec.entry_point),
                rl_libraries=format_rl_libraries(agents),
                presets=format_presets_rst(preset_map),
            )
        )

    rows.sort(key=lambda row: row.task_name)
    return rows


def _render_list_table_cell(value: str, indent: str = "      ") -> str:
    """Render one ``list-table`` cell, using a blank cell instead of a dash placeholder."""
    if not value:
        return f"{indent}-"
    return f"{indent}- {value}"


def render_comprehensive_list_table(rows: list[EnvironmentDocRow]) -> str:
    """Render the comprehensive environment ``list-table`` block as RST."""
    lines = [
        ".. list-table::",
        "    :widths: 18 17 10 22 33",
        "",
        "    * - **Task Name**",
        "      - **Inference Task Name**",
        "      - **Workflow**",
        "      - **RL Library**",
        "      - **Presets**",
    ]

    for row in rows:
        lines.extend(
            [
                f"    * - {row.task_name}",
                _render_list_table_cell(row.inference_task_name or ""),
                _render_list_table_cell(row.workflow),
                _render_list_table_cell(row.rl_libraries),
                _render_list_table_cell(row.presets),
            ]
        )

    return "\n".join(lines)


def patch_environments_rst(content: str, generated_table: str) -> str:
    """Replace the auto-generated comprehensive list section in *content*."""
    start = content.find(COMPREHENSIVE_LIST_START_MARKER)
    end = content.find(COMPREHENSIVE_LIST_END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "Could not find comprehensive list auto-generation markers in environments.rst. "
            f"Expected both '{COMPREHENSIVE_LIST_START_MARKER}' and '{COMPREHENSIVE_LIST_END_MARKER}'."
        )

    end += len(COMPREHENSIVE_LIST_END_MARKER)
    replacement = f"{COMPREHENSIVE_LIST_START_MARKER}\n\n{generated_table}\n\n{COMPREHENSIVE_LIST_END_MARKER}"
    return content[:start] + replacement + content[end:]


def _infer_algorithm(algo_suffix: str, entry_value: object, library: str) -> str:
    """Map a cfg-entry suffix and value to a display algorithm label."""
    if not algo_suffix:
        if library == "rl_games" and isinstance(entry_value, str) and "vision" in entry_value.lower():
            return "VISION"
        return "PPO"

    normalized = {
        "ppo": "PPO",
        "amp": "AMP",
        "ippo": "IPPO",
        "mappo": "MAPPO",
    }
    return normalized.get(algo_suffix.lower(), algo_suffix.upper())


def _algo_sort_key(algo: str) -> tuple[int, str]:
    """Sort PPO first, then alphabetically."""
    return (0 if algo == "PPO" else 1, algo)

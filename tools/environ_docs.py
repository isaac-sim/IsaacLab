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
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym

from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
from isaaclab_tasks.utils.preset_target import PresetTarget

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec

# Backend preset names that may be mirrored on untyped ``PresetCfg`` fields
# (for example contact-sensor or observation presets). These should only appear
# under their typed ``physics=`` or ``renderer=`` selectors, never ``presets=``.
_BACKEND_MIRROR_NAMES = frozenset(
    {
        "isaacsim_physx",
        "isaacsim_rtx",
        "newton_kamino",
        "newton_mjwarp",
        "newton_mjwarp_vbd_proxy",
        "ovphysx",
        "ovrtx",
        "physx",
        "rtx",
        *PresetTarget.all_legacy_aliases().keys(),
        *PresetTarget.all_legacy_aliases().values(),
    }
)

# RL libraries listed in a stable order across generated docs.
_RL_LIBRARY_ORDER = ("rl_games", "rsl_rl", "skrl", "sb3", "rlinf")

# Gym IDs excluded from the training list. The ``-Eval`` suffix marks dedicated
# evaluation variants (e.g. ``IsaacContrib-Assemble-Trocar-G129-Dex3-Eval``, an alias
# registered for RLinf eval configs) that should not appear as their own training row.
_EVAL_TASK_SUFFIXES = ("-Eval",)

# RL libraries not discoverable from Gym ``kwargs`` (e.g. RLinf YAML-based workflows).
RL_LIBRARY_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "IsaacContrib-Assemble-Trocar-G129-Dex3": {"rlinf": ["PPO"]},
}

# Marker comments that delimit the auto-generated section in environments.rst.
COMPREHENSIVE_LIST_START_MARKER = ".. START-AUTO-GENERATED: comprehensive-environment-list"
COMPREHENSIVE_LIST_END_MARKER = ".. END-AUTO-GENERATED: comprehensive-environment-list"
ENVIRONMENT_BROWSER_TASKS_START_MARKER = "// START-AUTO-GENERATED: environment-browser-task-rows"
ENVIRONMENT_BROWSER_TASKS_END_MARKER = "// END-AUTO-GENERATED: environment-browser-task-rows"

# Automatic backend-family aliases are intentionally omitted from environment
# tables. The concrete backend names make runtime requirements explicit and do
# not change meaning based on the rest of the selected configuration.
_OMITTED_SELECTOR_NAMES = {
    PresetTarget.PHYSICS: frozenset({"physx"}),
    PresetTarget.RENDERER: frozenset({"rtx"}),
    PresetTarget.DOMAIN: frozenset({"physx", "rtx"}),
}

_SELECTOR_LABELS = {
    PresetTarget.PHYSICS: "physics",
    PresetTarget.RENDERER: "renderer",
    PresetTarget.DOMAIN: "presets",
}


@dataclass(frozen=True)
class EnvironmentDocRow:
    """One row of the comprehensive environment list in ``environments.rst``."""

    task_name: str
    workflow: str
    rl_libraries: dict[str, list[str]]
    presets: dict[PresetTarget, list[str]] | None


def is_training_task(task_id: str) -> bool:
    """Return ``True`` when *task_id* is a training (non-inference) Isaac task."""
    if "Isaac" not in task_id:
        return False
    if any(task_id.endswith(suffix) for suffix in _EVAL_TASK_SUFFIXES):
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

    Some tasks mirror backend names on untyped ``PresetCfg`` fields (for
    example observation presets named ``physx`` / ``newton_mjwarp``, or
    contact-sensor presets named ``ovphysx``). Names that also exist on typed
    physics or renderer presets are selected through those targets and are
    filtered here. Backend names that appear only on untyped fields are omitted
    entirely because they are not meaningful ``presets=`` selectors for users.
    """
    typed_names = set(preset_map.get(PresetTarget.PHYSICS, [])) | set(preset_map.get(PresetTarget.RENDERER, []))
    domain_names: list[str] = []
    for name in preset_map.get(PresetTarget.DOMAIN, []):
        if name in typed_names or name in _BACKEND_MIRROR_NAMES:
            continue
        domain_names.append(name)
    return domain_names


def _selector_names_for_docs(
    preset_map: dict[PresetTarget, list[str]] | None,
) -> dict[PresetTarget, list[str]]:
    """Return concrete selector names grouped by their command-line target."""
    if preset_map is None:
        return {target: [] for target in _SELECTOR_LABELS}

    return {
        PresetTarget.PHYSICS: _filter_selector_names(PresetTarget.PHYSICS, preset_map.get(PresetTarget.PHYSICS, [])),
        PresetTarget.RENDERER: _filter_selector_names(PresetTarget.RENDERER, preset_map.get(PresetTarget.RENDERER, [])),
        PresetTarget.DOMAIN: _filter_selector_names(PresetTarget.DOMAIN, _domain_presets_for_docs(preset_map)),
    }


def format_presets_rst(preset_map: dict[PresetTarget, list[str]] | None) -> str:
    """Format concrete preset selectors for an RST ``list-table`` cell."""
    groups: list[tuple[str, list[str]]] = []
    selector_names = _selector_names_for_docs(preset_map)
    for target, label in _SELECTOR_LABELS.items():
        names = selector_names[target]
        if names:
            groups.append((f"{label}=", names))

    if not groups:
        return ""

    formatted_groups = [f"**{label}** {', '.join(f'``{name}``' for name in names)}" for label, names in groups]
    if len(formatted_groups) == 1:
        return formatted_groups[0]
    return "\n        ".join(
        f"| {group}" if index == 0 else f"  | {group}" for index, group in enumerate(formatted_groups)
    )


def _filter_selector_names(target: PresetTarget, names: list[str]) -> list[str]:
    """Return selector names after removing automatic backend-family aliases."""
    omitted = _OMITTED_SELECTOR_NAMES[target]
    return [name for name in names if name not in omitted]


def patch_curated_environment_tables(content: str, rows: list[EnvironmentDocRow]) -> str:
    """Update preset cells in the curated environment grid tables.

    Curated rows can contain more than one task ID. Their preset cell shows the
    union of the concrete selectors for those tasks; the comprehensive table
    remains the task-specific source of truth.

    Args:
        content: Complete ``environments.rst`` contents.
        rows: Environment rows collected from the Gym registry.

    Returns:
        Document contents with synchronized curated preset cells.
    """
    curated_end = content.find(COMPREHENSIVE_LIST_START_MARKER)
    if curated_end == -1:
        raise ValueError(f"Could not find '{COMPREHENSIVE_LIST_START_MARKER}' in environments.rst.")

    task_presets: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        selector_names = _selector_names_for_docs(row.presets)
        task_presets[row.task_name] = {label: set(selector_names[target]) for target, label in _SELECTOR_LABELS.items()}
    substitutions = _parse_task_link_substitutions(content[:curated_end])
    newline = "\r\n" if "\r\n" in content else "\n"
    curated_lines = content[:curated_end].splitlines()
    updated_lines: list[str] = []

    index = 0
    while index < len(curated_lines):
        line = curated_lines[index]
        if line.lstrip().startswith("+") and "Presets" in "\n".join(curated_lines[index : index + 8]):
            table_end = index
            while table_end < len(curated_lines) and curated_lines[table_end].lstrip().startswith(("+", "|")):
                table_end += 1
            updated_lines.extend(_patch_curated_grid_table(curated_lines[index:table_end], substitutions, task_presets))
            index = table_end
            continue

        updated_lines.append(line)
        index += 1

    curated = newline.join(updated_lines)
    if curated_lines:
        curated += newline
    return curated + content[curated_end:]


def _parse_task_link_substitutions(content: str) -> dict[str, str]:
    """Return ``{substitution_name: task_id}`` for environment links."""
    pattern = re.compile(r"^\.\. \|([^|]+)\| replace:: .*?`([^`<>]+?)\s*(?:<[^>]+>)?`\s*$", re.MULTILINE)
    return {match.group(1): match.group(2).strip() for match in pattern.finditer(content)}


def _patch_curated_grid_table(
    table_lines: list[str], substitutions: dict[str, str], task_presets: dict[str, dict[str, set[str]]]
) -> list[str]:
    """Synchronize one curated RST grid table with collected preset rows."""
    if not table_lines:
        return table_lines

    column_boundaries = [index for index, char in enumerate(table_lines[0]) if char == "+"]
    if len(column_boundaries) < 3:
        return table_lines

    row_blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(table_lines):
        if not table_lines[index].lstrip().startswith("|"):
            index += 1
            continue
        row_start = index
        while index < len(table_lines) and table_lines[index].lstrip().startswith("|"):
            index += 1
        row_blocks.append((row_start, index))

    parsed_blocks = {
        block: _parse_grid_row(table_lines[start:end], column_boundaries)
        for block, (start, end) in enumerate(row_blocks)
    }
    header = parsed_blocks.get(0, [])
    environment_column = next(
        (column for column, values in enumerate(header) if "Environment ID" in " ".join(values)), None
    )
    preset_column = next((column for column, values in enumerate(header) if "Presets" in " ".join(values)), None)
    if environment_column is None or preset_column is None:
        return table_lines

    replacement_cells: dict[int, dict[str, set[str]]] = {}
    for block, cells in parsed_blocks.items():
        if block == 0:
            continue
        task_ids = []
        for substitution in re.findall(r"\|([^|]+)\|", " ".join(cells[environment_column])):
            task_id = substitutions.get(substitution)
            if task_id in task_presets:
                task_ids.append(task_id)
        if not task_ids:
            continue
        replacement_cells[block] = {
            label: set().union(*(task_presets[task_id][label] for task_id in task_ids))
            for label in _SELECTOR_LABELS.values()
        }

    original_widths = [end - start - 1 for start, end in zip(column_boundaries, column_boundaries[1:])]
    preset_width = original_widths[preset_column]
    longest_literal = max(
        (len(f"``{name}``") for groups in replacement_cells.values() for names in groups.values() for name in names),
        default=0,
    )
    updated_widths = list(original_widths)
    updated_widths[preset_column] = max(preset_width, longest_literal + 3)

    rendered_cells = {
        block: _render_curated_preset_cell(groups, updated_widths[preset_column] - 2)
        for block, groups in replacement_cells.items()
    }

    block_by_start = {start: block for block, (start, _) in enumerate(row_blocks)}
    output: list[str] = []
    index = 0
    while index < len(table_lines):
        line = table_lines[index]
        if line.lstrip().startswith("+"):
            output.append(_resize_grid_border(line, updated_widths))
            index += 1
            continue

        block = block_by_start.get(index)
        if block is None:
            output.append(line)
            index += 1
            continue

        start, end = row_blocks[block]
        cells = parsed_blocks[block]
        if block in rendered_cells:
            cells[preset_column] = rendered_cells[block]
        output.extend(_render_grid_row(cells, updated_widths, table_lines[start][: column_boundaries[0]]))
        index = end

    return output


def _parse_grid_row(lines: list[str], boundaries: list[int]) -> list[list[str]]:
    """Parse the cells of one multi-line RST grid-table row."""
    return [[line[start + 1 : end].strip() for line in lines] for start, end in zip(boundaries, boundaries[1:])]


def _render_curated_preset_cell(groups: dict[str, set[str]], content_width: int) -> list[str]:
    """Render concrete selectors into wrapped grid-cell lines."""
    lines: list[str] = []
    for label in _SELECTOR_LABELS.values():
        names = sorted(groups[label])
        if not names:
            continue
        pieces = [f"``{name}``{',' if index < len(names) - 1 else ''}" for index, name in enumerate(names)]
        current = f"**{label}=**"
        for piece in pieces:
            candidate = f"{current} {piece}"
            if len(candidate) <= content_width:
                current = candidate
            else:
                lines.append(current)
                current = piece
        lines.append(current)
    return lines or [""]


def _resize_grid_border(line: str, updated_widths: list[int]) -> str:
    """Resize the final preset column of one RST grid-table border."""
    indent = line[: len(line) - len(line.lstrip())]
    stripped = line.strip()
    fills = [segment[0] if segment else "-" for segment in stripped[1:-1].split("+")]
    segments = [fill * width for fill, width in zip(fills, updated_widths)]
    return f"{indent}+{'+'.join(segments)}+"


def _render_grid_row(cells: list[list[str]], widths: list[int], indent: str) -> list[str]:
    """Render one parsed RST grid-table row using the requested widths."""
    for cell in cells:
        while len(cell) > 1 and not cell[-1]:
            cell.pop()
    height = max((len(cell) for cell in cells), default=1)
    output = []
    for line_index in range(height):
        values = [cell[line_index] if line_index < len(cell) else "" for cell in cells]
        rendered = [f" {value:<{width - 2}} " for value, width in zip(values, widths)]
        output.append(f"{indent}|{'|'.join(rendered)}|")
    return output


def get_workflow(entry_point: str) -> str:
    """Return the human-readable workflow label for a Gym entry point."""
    if "ManagerBasedRLEnv" in entry_point:
        return "Manager Based"
    return "Direct"


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

    rows: list[EnvironmentDocRow] = []

    for spec in specs:
        if not is_training_task(spec.id) or spec.kwargs.get("deprecated"):
            continue

        preset_map = enumerate_task_presets(spec.id)
        if preset_map is not None:
            preset_map = dict(preset_map)
            preset_map[PresetTarget.PHYSICS] = _physics_names_for_docs(spec.id, preset_map)
        agents = apply_rl_library_overrides(spec.id, parse_rl_libraries_from_kwargs(spec.kwargs))

        rows.append(
            EnvironmentDocRow(
                task_name=spec.id,
                workflow=get_workflow(spec.entry_point),
                rl_libraries=agents,
                presets=preset_map,
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
        "    :widths: 22 12 30 36",
        "",
        "    * - **Task Name**",
        "      - **Workflow**",
        "      - **RL Library**",
        "      - **Presets**",
    ]

    for row in rows:
        lines.extend(
            [
                f"    * - {row.task_name}",
                _render_list_table_cell(row.workflow),
                _render_list_table_cell(format_rl_libraries(row.rl_libraries)),
                _render_list_table_cell(format_presets_rst(row.presets)),
            ]
        )

    return "\n".join(lines)


def render_environment_browser_task_rows(rows: list[EnvironmentDocRow]) -> str:
    """Render concrete core-task selectors for the environment browser."""
    lines = ["        const taskRows = ["]
    for row in rows:
        if not row.task_name.startswith("Isaac-"):
            continue
        selectors = _selector_names_for_docs(row.presets)
        values = (
            row.task_name,
            ",".join(library for library in _RL_LIBRARY_ORDER if library in row.rl_libraries),
            ",".join(sorted(selectors[PresetTarget.PHYSICS])),
            ",".join(sorted(selectors[PresetTarget.RENDERER])),
            ",".join(sorted(selectors[PresetTarget.DOMAIN])),
        )
        rendered_values = ", ".join(json.dumps(value) for value in values)
        lines.append(f"            [{rendered_values}],")
    lines.append("        ];")
    return "\n".join(lines)


def patch_environment_browser_javascript(content: str, generated_rows: str) -> str:
    """Replace the auto-generated task rows in ``environment-browser.js``."""
    start = content.find(ENVIRONMENT_BROWSER_TASKS_START_MARKER)
    end_marker_start = content.find(ENVIRONMENT_BROWSER_TASKS_END_MARKER)
    if start == -1 or end_marker_start == -1 or end_marker_start < start:
        raise ValueError(
            "Could not find environment-browser task markers. "
            f"Expected both '{ENVIRONMENT_BROWSER_TASKS_START_MARKER}' and "
            f"'{ENVIRONMENT_BROWSER_TASKS_END_MARKER}'."
        )

    generated_region = content[start + len(ENVIRONMENT_BROWSER_TASKS_START_MARKER) : end_marker_start]
    task_rows_start = generated_region.find("const taskRows = [")
    task_rows_end = generated_region.find("];", task_rows_start)
    if task_rows_start == -1 or task_rows_end == -1 or generated_region[task_rows_end + 2 :].strip():
        raise ValueError("Environment-browser task markers must contain only the generated taskRows array.")

    end = end_marker_start + len(ENVIRONMENT_BROWSER_TASKS_END_MARKER)
    indent = content[content.rfind("\n", 0, start) + 1 : start]
    replacement = (
        f"{ENVIRONMENT_BROWSER_TASKS_START_MARKER}\n{generated_rows}\n{indent}{ENVIRONMENT_BROWSER_TASKS_END_MARKER}"
    )
    return content[:start] + replacement + content[end:]


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

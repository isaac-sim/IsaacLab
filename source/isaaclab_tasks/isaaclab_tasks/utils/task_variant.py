# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discovery and validation for task presets that require specific agent configs."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
from typing import Any

import gymnasium as gym

_AGENT_LIBRARIES = ("rl_games", "rsl_rl", "skrl", "sb3", "rlinf")


@dataclasses.dataclass(frozen=True)
class TaskVariantCfg:
    """Compatibility between a task's domain presets and registered agent configs.

    Add this object to a task's Gym registration under the ``"task_variant_cfg"``
    key when selecting a domain preset changes the observation or action interface
    expected by the agent. Physics and renderer presets do not need to be listed
    unless they also change that interface.

    Args:
        default_preset: Effective domain preset when no ``presets=`` selector is
            passed.
        agents: Registered agent config entry-point keys and the domain presets
            each one supports.
    """

    @dataclasses.dataclass(frozen=True)
    class AgentCfg:
        """Compatibility declared for one registered agent config.

        Args:
            preset_names: Domain preset names supported by the agent config.
            description: Short user-facing explanation of the agent config.
        """

        preset_names: tuple[str, ...]
        description: str = ""

    default_preset: str
    agents: dict[str, AgentCfg]


def enumerate_task_agents(task_name: str, agent_library: str | None = None) -> dict[str, Any]:
    """Return agent config entry points registered for a task.

    Args:
        task_name: Gymnasium task ID.
        agent_library: Optional RL-library prefix such as ``"rl_games"`` or
            ``"rsl_rl"``.

    Returns:
        Mapping from the accepted ``--agent`` value to its registered config
        entry point, sorted by key.
    """
    spec = gym.spec(task_name.split(":")[-1])
    prefix = f"{agent_library}_" if agent_library else None
    agents = {
        key: value
        for key, value in spec.kwargs.items()
        if key.endswith("_cfg_entry_point")
        and key != "env_cfg_entry_point"
        and (prefix is None or key.startswith(prefix))
    }
    return dict(sorted(agents.items()))


def enumerate_task_variants(task_name: str, agent_library: str | None = None) -> dict[str, Any]:
    """Return machine-readable preset and agent discovery for a task.

    Args:
        task_name: Gymnasium task ID.
        agent_library: Optional RL-library prefix used to filter agent configs.

    Returns:
        Dictionary containing typed selectors, registered agent config names,
        and declared preset compatibility.

    Raises:
        ValueError: If task compatibility metadata references a preset or agent
            config that is not registered.
    """
    from isaaclab_tasks.utils.preset_cli import enumerate_task_presets

    task = task_name.split(":")[-1]
    # ``load_cfg_from_registry`` announces config paths on stdout. Discovery is
    # also a machine-readable CLI path, so keep those informational messages
    # out of table/JSON output.
    with contextlib.redirect_stdout(io.StringIO()):
        preset_map = enumerate_task_presets(task)
    if preset_map is None:
        raise ValueError(f"Could not enumerate presets for task '{task}'.")

    selectors = {target.value: names for target, names in preset_map.items()}
    registered_agents = enumerate_task_agents(task, agent_library)
    variant_cfg = _get_task_variant_cfg(task)
    if variant_cfg is not None:
        _validate_task_variant_cfg(task, variant_cfg, selectors, enumerate_task_agents(task))

    default_agent = f"{agent_library}_cfg_entry_point" if agent_library else None
    agents = []
    for name in registered_agents:
        compatibility = variant_cfg.agents.get(name) if variant_cfg else None
        agents.append(
            {
                "name": name,
                "default": name == default_agent,
                "compatible_presets": sorted(compatibility.preset_names) if compatibility else None,
                "description": compatibility.description if compatibility else "",
            }
        )

    return {
        "task": task,
        "agent_library": agent_library,
        "default_preset": variant_cfg.default_preset if variant_cfg else None,
        "selectors": selectors,
        "agents": agents,
    }


def format_task_variants(task_name: str, agent_library: str | None = None, output_format: str = "table") -> str:
    """Format task variant discovery for terminal or automation use.

    Args:
        task_name: Gymnasium task ID.
        agent_library: Optional RL-library prefix used to filter agent configs.
        output_format: Output format, either ``"table"`` or ``"json"``.

    Returns:
        Formatted task variant discovery.

    Raises:
        ValueError: If :paramref:`output_format` is unsupported.
    """
    variants = enumerate_task_variants(task_name, agent_library)
    if output_format == "json":
        return json.dumps(variants, indent=2)
    if output_format != "table":
        raise ValueError(f"Unsupported task variant output format: {output_format!r}.")

    lines = [f"Task variants for {variants['task']}"]
    if agent_library:
        lines[0] += f" ({agent_library})"
    lines.append("")
    lines.append("Selectors:")
    for selector, names in variants["selectors"].items():
        lines.append(f"  {selector}= {', '.join(names) if names else '(none)'}")

    lines.extend(["", "Agent configurations:"])
    if not variants["agents"]:
        lines.append("  (none)")
    for agent in variants["agents"]:
        default = " (default)" if agent["default"] else ""
        lines.append(f"  --agent {agent['name']}{default}")
        compatible = agent["compatible_presets"]
        if compatible is None:
            lines.append("    compatible presets: not declared")
        else:
            lines.append(f"    compatible presets: {', '.join(compatible)}")
        if agent["description"]:
            lines.append(f"    {agent['description']}")
    return "\n".join(lines)


def validate_task_variant(task_name: str, agent_entry: str | None, selected_presets: list[str]) -> None:
    """Validate a selected agent against domain-preset compatibility metadata.

    Tasks without :class:`TaskVariantCfg` metadata are left unchanged. Typed
    physics and renderer names are ignored unless they also appear in the
    declared domain-preset compatibility sets.

    Args:
        task_name: Gymnasium task ID.
        agent_entry: Selected agent config entry-point key.
        selected_presets: Preset names parsed from all CLI selectors.

    Raises:
        ValueError: If the selected agent and effective domain preset are not
            compatible.
    """
    if agent_entry is None:
        return
    variant_cfg = _get_task_variant_cfg(task_name)
    if variant_cfg is None:
        return

    declared_presets = {
        preset_name for agent_cfg in variant_cfg.agents.values() for preset_name in agent_cfg.preset_names
    }
    selected_domain_presets = [name for name in selected_presets if name in declared_presets]
    effective_presets = selected_domain_presets or [variant_cfg.default_preset]
    agent_cfg = variant_cfg.agents.get(agent_entry)
    allowed_presets = set(agent_cfg.preset_names) if agent_cfg else set()
    incompatible = sorted(set(effective_presets) - allowed_presets)
    if not incompatible:
        return

    selected_library = next(
        (library for library in _AGENT_LIBRARIES if agent_entry.startswith(f"{library}_")),
        None,
    )
    compatible_agents = sorted(
        name
        for name, candidate in variant_cfg.agents.items()
        if set(effective_presets).issubset(candidate.preset_names)
        and (selected_library is None or name.startswith(f"{selected_library}_"))
    )
    preset_arg = ",".join(effective_presets)
    suggestions = "\n".join(f"  --agent {name} presets={preset_arg}" for name in compatible_agents)
    if not suggestions:
        suggestions = "  (no registered agent supports this preset combination)"
    raise ValueError(
        f"Agent config '{agent_entry}' is incompatible with presets={preset_arg} for task '{task_name}'.\n"
        f"Compatible command arguments:\n{suggestions}\n"
        "Run the command with --help or --list_variants table to inspect supported combinations."
    )


def _get_task_variant_cfg(task_name: str) -> TaskVariantCfg | None:
    """Return and type-check task variant metadata from the Gym registry."""
    task = task_name.split(":")[-1]
    cfg = gym.spec(task).kwargs.get("task_variant_cfg")
    if cfg is not None and not isinstance(cfg, TaskVariantCfg):
        raise TypeError(f"Task '{task}' registered task_variant_cfg={cfg!r}, expected TaskVariantCfg.")
    return cfg


def _validate_task_variant_cfg(
    task_name: str,
    variant_cfg: TaskVariantCfg,
    selectors: dict[str, list[str]],
    registered_agents: dict[str, Any],
) -> None:
    """Check compatibility metadata against live preset and agent discovery."""
    domain_presets = set(selectors.get("presets", []))
    declared_presets = {
        preset_name for agent_cfg in variant_cfg.agents.values() for preset_name in agent_cfg.preset_names
    }
    unknown_presets = declared_presets - domain_presets
    if unknown_presets:
        raise ValueError(
            f"Task '{task_name}' compatibility metadata references unknown domain presets: "
            f"{', '.join(sorted(unknown_presets))}."
        )
    if variant_cfg.default_preset not in declared_presets:
        raise ValueError(
            f"Task '{task_name}' default compatibility preset '{variant_cfg.default_preset}' is not assigned to an"
            " agent."
        )
    unknown_agents = set(variant_cfg.agents) - set(registered_agents)
    if unknown_agents:
        raise ValueError(
            f"Task '{task_name}' compatibility metadata references unknown agent configs: "
            f"{', '.join(sorted(unknown_agents))}."
        )


from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.kit.app

from isaaclab.managers.command_manager import CommandTerm, CommandManager
from .parkour_manager_term_cfg import ParkourTermCfg

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

"""
Parkour Manager is dealing with goal heading position
It is similar to a CommandMangner which is a handling Position Command
"""

class ParkourTerm(CommandTerm): 
    def __init__(self, cfg: ParkourTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env) 

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value).item()
            metric_value[env_ids] = 0.0

        self._resample(env_ids)

        return extras
    
    def _resample(self, env_ids: Sequence[int]):
        if len(env_ids) != 0:
            self._resample_command(env_ids)

    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        self._update_command()
        self._update_metrics()

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command generator has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code
    
    def __call__(self):
        pass 


    
class ParkourManager(CommandManager):
    _env: ParkourManagerBasedRLEnv
    def __init__(self, cfg: object, env: ParkourManagerBasedRLEnv):        
        super().__init__(cfg, env) 

    def __call__(self):
        for term in self._terms.values():
            term()

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, ParkourTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ParkourTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ParkourTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ParkourType.")
            # add class to dict
            self._terms[term_name] = term


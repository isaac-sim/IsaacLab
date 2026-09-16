# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTermCfg

if TYPE_CHECKING:
    from .recorders import (
        InitialStateRecorder,
        PostStepProcessedActionsRecorder,
        PostStepStatesRecorder,
        PreStepActionsRecorder,
        PreStepFlatPolicyObservationsRecorder,
    )

##
# State recorders.
##


@dataclass
class InitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type["InitialStateRecorder"] | str = "{DIR}.recorders:InitialStateRecorder"


@dataclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type["PostStepStatesRecorder"] | str = "{DIR}.recorders:PostStepStatesRecorder"


@dataclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type["PreStepActionsRecorder"] | str = "{DIR}.recorders:PreStepActionsRecorder"


@dataclass
class PreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type["PreStepFlatPolicyObservationsRecorder"] | str = (
        "{DIR}.recorders:PreStepFlatPolicyObservationsRecorder"
    )


@dataclass
class PostStepProcessedActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the post step processed actions recorder term."""

    class_type: type["PostStepProcessedActionsRecorder"] | str = "{DIR}.recorders:PostStepProcessedActionsRecorder"


##
# Recorder manager configurations.
##


@dataclass
class ActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()

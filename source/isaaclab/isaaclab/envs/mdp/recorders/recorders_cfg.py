# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders

##
# State recorders.
##


@configclass
class InitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type[RecorderTerm] = recorders.InitialStateRecorder


@configclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepStatesRecorder


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type[RecorderTerm] = recorders.PreStepActionsRecorder


@configclass
class PreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = recorders.PreStepFlatPolicyObservationsRecorder


##
# Recorder manager configurations.
##


@configclass
class ActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()

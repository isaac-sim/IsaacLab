# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from rai.eval_sim.ros_manager.subscribers_cfg import (
    ResetSubscriberCfg,
    SimulationParametersSubscriberCfg,
)


@configclass
class RosManagerCfg:
    reset_term: ResetSubscriberCfg = ResetSubscriberCfg()
    simulation_parameters = SimulationParametersSubscriberCfg(topic="/simulation_parameters")

    """Default ResetSubscriber term."""
    lockstep_timeout: float | None = None
    """Seconds to wait. Block forever if None or negative. Don’t wait if 0."""
    use_sim_time: bool = True
    """If true, use the environment's sim time when filling the timestamp. Else, use the ROS node time."""

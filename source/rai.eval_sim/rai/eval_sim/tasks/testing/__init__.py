# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from . import anymal_d_testing_env_cfg as test_env_cfgs
from . import anymal_d_testing_ros_manager_cfg as test_ros_cfgs

TESTING_CFGS = {
    "BaseLine": (test_env_cfgs.AnymalDBaseLineEnvCfg, test_ros_cfgs.AnymalDBaseLineRosCfg),
    "BaseLine2": (test_env_cfgs.AnymalDBaseLineEnvCfg, test_ros_cfgs.AnymalDBaseLineRosCfg),
    "+PDGainsSub": (test_env_cfgs.AnymalDBaseLineEnvCfg, test_ros_cfgs.AnymalDPlusPDGainsRMcfg),
    "+LinkPoseObsPub": (test_env_cfgs.AnymalDPlusLinkPoseObsCfg, test_ros_cfgs.AnymalDPlusLinkPoseObsCfg),
    "+TwistObsPub": (test_env_cfgs.AnymalDPlusTwistObsCfg, test_ros_cfgs.AnymalDPlusTwistObsCfg),
    "+ProjGravityObsPub": (test_env_cfgs.AnymalDPlusProjGravEnvCfg, test_ros_cfgs.AnymalDPlusProjGravObsRMCfg),
    "+ContactForcePub": (test_env_cfgs.AnymalDPlusContactEnvCfg, test_ros_cfgs.AnymalDPlusContactRMCfg),
    "+ImuObsPub": (test_env_cfgs.AnymalDPlusImuEnvCfg, test_ros_cfgs.AnymalDPlusImuRMCfg),
    "+GridMapPub": (test_env_cfgs.AnymalDPlusGridMapEnvCfg, test_ros_cfgs.AnymalDPlusGridMapRMCfg),
    "+HeightScanObsPub": (test_env_cfgs.AnymalDPlusHeightScanEnvCfg, test_ros_cfgs.AnymalDPlusHeightScanRMCfg),
    "+WrenchObsPub": (test_env_cfgs.AnymalDPlusWrenchEnvCfg, test_ros_cfgs.AnymalDPlusWrenchRMCfg),
    "+AllWrenchObsPub": (
        test_env_cfgs.AnymalDPlusJointReactionWrenchEnvCfg,
        test_ros_cfgs.AnymalDPlusJointReactionWrenchObsPublisherRMCfg,
    ),
}

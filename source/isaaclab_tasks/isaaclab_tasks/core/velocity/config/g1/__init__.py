# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_env_cfg:G129DofRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-RealAnkle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_env_cfg:G129DofRoughRealAnkleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-RealTorque",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_env_cfg:G129DofRoughRealTorqueEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-WBC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_wbc_env_cfg:G129DofRoughWbcEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-NoHeight",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughNoHeightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-HipL2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughHipL2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-OldAsset-Height",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G1RoughOldAssetHeightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-HipL2-AirTime050",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughHipL2AirTime050EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-HipL2-AirTime100",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughHipL2AirTime100EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-HipL2-AirTime150",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughHipL2AirTime150EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-Imbalance05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughImbalance05EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-Imbalance20",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughImbalance20EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-Imbalance50",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughImbalance50EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-HipL2-AirTime4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughHipL2AirTime4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-HipL2-AirTime8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughHipL2AirTime8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-HipL2-AirTime16",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_posture_env_cfg:G129DofRoughHipL2AirTime16EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-RealAnkle-AirTime4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_env_cfg:G129DofRoughRealAnkleAirTime4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-RealAnkle-AirTime8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_env_cfg:G129DofRoughRealAnkleAirTime8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)





gym.register(
    id="Isaac-Velocity-Flat-G1-29Dof",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_29dof_env_cfg:G129DofFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-AirTime100-DR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_dr_env_cfg:G129DofRoughAirTime100DREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-AirTime100-DR-History",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_dr_env_cfg:G129DofRoughAirTime100DRHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-AirTime100-Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_distill_env_cfg:G129DofRoughAirTime100RobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-AirTime100-Distill",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_distill_env_cfg:G129DofRoughAirTime100DistillEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G129DofRoughAirTime100DistillationRunnerCfg"
        ),
    },
)


for _power_arm, _power_cls in (
    ("Power1", "G129DofRoughAirTime100Power1EnvCfg"),
    ("Power2", "G129DofRoughAirTime100Power2EnvCfg"),
    ("Power3", "G129DofRoughAirTime100Power3EnvCfg"),
    ("Power4", "G129DofRoughAirTime100Power4EnvCfg"),
    ("Power5", "G129DofRoughAirTime100Power5EnvCfg"),
):
    gym.register(
        id=f"Isaac-Velocity-Rough-G1-29Dof-AirTime100-{_power_arm}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rough_29dof_power_env_cfg:{_power_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        },
    )


for _waist_arm, _waist_cls in (
    ("Waist1", "G129DofRoughAirTime100DRWaist1EnvCfg"),
    ("Waist3", "G129DofRoughAirTime100DRWaist3EnvCfg"),
):
    gym.register(
        id=f"Isaac-Velocity-Rough-G1-29Dof-AirTime100-DR-{_waist_arm}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rough_29dof_dr_env_cfg:{_waist_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        },
    )


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-AirTime100-History",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_dr_env_cfg:G129DofRoughAirTime100HistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


for _sym_arm, _sym_cls in (
    ("Sym1", "G129DofRoughAirTime100Sym1EnvCfg"),
    ("Sym2", "G129DofRoughAirTime100Sym2EnvCfg"),
    ("Sym3", "G129DofRoughAirTime100Sym3EnvCfg"),
):
    gym.register(
        id=f"Isaac-Velocity-Rough-G1-29Dof-AirTime100-{_sym_arm}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rough_29dof_symmetry_env_cfg:{_sym_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        },
    )


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-AirTime100-Waist1-Power2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_29dof_power_env_cfg:G129DofRoughWaist1Power2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-AirTime100-Robust-Waist",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rough_29dof_distill_env_cfg:G129DofRoughAirTime100RobustWaistEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-29Dof-AirTime100-Waist1-Power2-HipPitch",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rough_29dof_power_env_cfg:G129DofRoughWaist1Power2HipPitchEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

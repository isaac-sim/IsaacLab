# Environment Templates

## Full Manager-Based Environment Template

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Import your robot config
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene: ground + robot + lights."""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """What the policy controls."""
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0,
    )


@configclass
class ObservationsCfg:
    """What the policy sees."""
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset randomization."""
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.25, 0.25), "velocity_range": (-0.25, 0.25)},
    )


@configclass
class RewardsCfg:
    """Reward signal design."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # TODO: Add task-specific rewards


@configclass
class TerminationsCfg:
    """Episode end conditions."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # TODO: Add task-specific terminations


@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    """Main environment config."""
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
```

## SB3 Agent Config Template (`sb3_ppo_cfg.yaml`)

```yaml
seed: 42
n_timesteps: 1000000
policy: 'MlpPolicy'
n_steps: 16
batch_size: 4096
n_epochs: 20
learning_rate: 3e-4
clip_range: 0.2
gamma: 0.99
gae_lambda: 0.95
ent_coef: 0.01
vf_coef: 1.0
max_grad_norm: 1.0
policy_kwargs:
  activation_fn: 'nn.ELU'
  net_arch: [32, 32]
device: "cuda:0"
```

## SKRL Agent Config Template (`skrl_ppo_cfg.yaml`)

```yaml
seed: 42
models:
  separate: False
  policy:
    class: GaussianMixin
    clip_actions: False
    clip_log_std: True
    min_log_std: -20.0
    max_log_std: 2.0
    initial_log_std: 0.0
    network:
      - name: net
        input: OBSERVATIONS
        layers: [32, 32]
        activations: elu
    output: ACTIONS
  value:
    class: DeterministicMixin
    clip_actions: False
    network:
      - name: net
        input: OBSERVATIONS
        layers: [32, 32]
        activations: elu
    output: ONE
memory:
  class: RandomMemory
  memory_size: -1
agent:
  class: PPO
  rollouts: 16
  learning_epochs: 8
  mini_batches: 8
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 3.0e-04
  learning_rate_scheduler: KLAdaptiveLR
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.008
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  entropy_loss_scale: 0.0
  value_loss_scale: 2.0
  experiment:
    directory: "my_task"
    experiment_name: ""
    write_interval: auto
    checkpoint_interval: auto
trainer:
  class: SequentialTrainer
  timesteps: 2400
  environment_info: log
```

## Gymnasium Registration Template (`__init__.py`)

```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env_cfg:MyEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

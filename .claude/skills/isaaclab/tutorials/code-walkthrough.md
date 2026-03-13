# CartPole Code Walkthrough

## Robot Asset
**File**: `source/isaaclab_assets/isaaclab_assets/robots/cartpole.py`

```python
CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),  # 2m above ground
        joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,
            stiffness=0.0,   # Pure effort control (no spring)
            damping=10.0,    # Cart friction
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,     # Free-swinging pole (no friction)
        ),
    },
)
```

## Scene Configuration

```python
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

## Actions

```python
@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,  # Policy output [-1,1] multiplied by 100 for force
    )
```

The policy outputs a scalar in [-1, 1]. Multiplied by `scale=100.0`, giving force range [-100, 100] N.

## Observations

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)  # 2 values
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)  # 2 values
        # Total: 4D observation vector
```

## Rewards (5 terms)

```python
@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

**Reward breakdown:**
| Term | Weight | Purpose |
|------|--------|---------|
| `alive` | +1.0 | Reward for each step survived |
| `terminating` | -2.0 | Penalty when episode ends by failure |
| `pole_pos` | -1.0 | L2 penalty: keep pole angle at 0 (upright) |
| `cart_vel` | -0.01 | L1 penalty: minimize cart movement |
| `pole_vel` | -0.005 | L1 penalty: minimize pole wobble |

**How `joint_pos_target_l2` works:**
```python
def joint_pos_target_l2(env, target, asset_cfg):
    asset = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)
```

## Terminations

```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "bounds": (-3.0, 3.0)},
    )
```

## Environment Config

```python
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2          # 2 physics steps per policy step
        self.episode_length_s = 5    # 5 second episodes
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120       # 120 Hz physics
        self.sim.render_interval = self.decimation
```

**Derived values:**
- Control frequency: 120 Hz / 2 = 60 Hz
- Max episode steps: 5s * 60 Hz = 300 steps
- Env spacing: 4.0 units between each CartPole instance

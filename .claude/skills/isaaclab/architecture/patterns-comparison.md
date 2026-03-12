# Manager-Based vs Direct: Detailed Comparison

## Manager-Based

**Class hierarchy**: `ManagerBasedEnvCfg` -> `ManagerBasedRLEnvCfg`

**Advantages:**
- Modular: swap reward terms without touching other code
- Reusable: built-in MDP functions work across tasks
- Configurable: change behavior by editing config, not code
- Observable: individual reward terms tracked in TensorBoard
- Extensible: custom terms via base classes (`RewardTermBase`, etc.)

**Config classes:**
- `ObservationsCfg` with `ObsGroup` -> `ObsTerm`
- `ActionsCfg` with action term configs
- `RewardsCfg` with `RewTerm` (func + weight + params)
- `TerminationsCfg` with `DoneTerm`
- `EventCfg` with `EventTerm` (mode: "reset" | "interval" | "startup")
- `CommandsCfg` (optional: goal commands for reaching, tracking)
- `CurriculumCfg` (optional: progressive difficulty)
- `RecorderCfg` (optional: data logging)

**Extension points** (custom terms inherit from):
- `ActionTermBase`, `ObservationTermBase`, `RewardTermBase`
- `TerminationTermBase`, `EventTermBase`, `CommandTermBase`
- `CurriculumTermBase`, `RecorderTermBase`

## Direct

**Class hierarchy**: `DirectRLEnv` (single agent), `DirectMARLEnv` (multi-agent)

**Advantages:**
- Simpler mental model: all logic in one place
- Slightly faster: no manager overhead
- Full control: no framework abstractions
- Multi-agent: `DirectMARLEnv` supports MARL natively

**Required overrides:**
```python
def _setup_scene(self):
    """Create articulations, rigid bodies, sensors, terrains."""

def _pre_physics_step(self, actions: torch.Tensor):
    """Process actions before simulation step."""

def _apply_action(self):
    """Apply processed actions to the simulation."""

def _get_observations(self) -> dict:
    """Return {"policy": tensor} observation dict."""

def _get_rewards(self) -> torch.Tensor:
    """Compute and return reward tensor (num_envs,)."""

def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (terminated, truncated) boolean tensors."""

def _reset_idx(self, env_ids: torch.Tensor):
    """Reset specific environment indices."""
```

## Side-by-Side

| Aspect | Manager-Based | Direct |
|--------|---------------|--------|
| Reward modification | Edit config weights | Edit function code |
| Adding observations | Add ObsTerm to config | Modify tensor concatenation |
| Reset randomization | EventTerm with mode="reset" | Code in _reset_idx |
| Observation groups | Multiple (policy, critic) | Manual dict construction |
| Reward tracking | Automatic per-term logging | Manual logging |
| Multi-agent | Not supported | DirectMARLEnv |
| Performance | Slight overhead | Fastest |

## Gymnasium Registration

Both patterns register identically:
```python
gym.register(
    id="Isaac-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # or DirectRLEnv
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "my_module:MyEnvCfg",
        "sb3_cfg_entry_point": "my_module.agents:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": "my_module.agents:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": "my_module.agents:MyRunnerCfg",
    },
)
```

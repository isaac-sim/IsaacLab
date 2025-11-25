# MP Usage in IsaacLab

- **Enable MP mode**: Register an MP variant of a step env using `register_mp_env` from `isaaclab_tasks.utils.mp`. Example for box pushing:
  ```python
  from isaaclab_tasks.manager_based.box_pushing.mp_wrapper import BoxPushingMPWrapper
  from isaaclab_tasks.utils.mp import register_mp_env, parse_env_cfg
  mp_id = register_mp_env(
      mp_id="Isaac_MP/Box-Pushing-Dense-ProDMP-Franka-v0",
      base_id="Isaac-Box-Pushing-Dense-step-Franka-v0",
      mp_wrapper_cls=BoxPushingMPWrapper,
      mp_type="ProDMP",
      device="cuda:0",
  )
  env = gym.make(mp_id, cfg=parse_env_cfg("Isaac-Box-Pushing-Dense-step-Franka-v0", device="cuda:0"))
  ```
- **Configure MP hyperparameters**: Override `mp_config` on your task-specific MP wrapper (basis count, gains, device) or pass `mp_config_override` into `register_mp_env`. All components (phase/basis/trajectory/controller) honor `device` and keep tensors on GPU.
- **Action bounds**: Implement `action_bounds` on your MP wrapper to expose meaningful step-action limits; the MP wrapper clamps using these bounds before env stepping.
- **Context observations**: Use `ContextObsWrapper` with a boolean mask to expose the MP context slice as a flat `Box` for RL libraries.
- **Training**: For MP-based training, set `num_steps_per_env=1` in your runner configs and use the MP env id (e.g., `Isaac_MP/...`). Switch back to step-based training by using the original step env id.
- **Example**: `scripts/environments/random_agent_mp.py` shows a minimal MP rollout with random MP actions (no fancy_gym dependency).

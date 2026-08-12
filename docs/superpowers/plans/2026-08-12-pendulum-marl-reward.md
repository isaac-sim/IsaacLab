---
orphan: true
---

# Pendulum MARL Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select and ship a shared team reward that makes `Isaac-Pendulum-MARL-Direct` train robustly across seeds while preserving its strict success metric.

**Architecture:** Temporarily expose two pure Torch team-reward helpers in the existing direct MARL environment, activate and benchmark them one at a time, then delete the losing helper. The production environment gives both agents the exact same reward tensor; actions, observations, resets, terminations, and success tracking stay unchanged.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab `DirectMARLEnv`, PhysX, skrl PPO, uv, jq, pre-commit.

## Global Constraints

- Keep `DirectMARLEnv`; do not add a single-agent environment.
- Keep success at both physical links within 15 degrees for the final 1 second of a timeout-completed episode.
- Do not add pendulum-specific test files.
- Use the same team reward for the `cart` and `pendulum` agents.
- Scale both experimental reward formulations by `env.step_dt`.
- Run PhysX/skrl training with 4,096 environments and seeds 42, 43, and 44.
- Rank candidates by worst-seed final-50 success, then aggregate success, collapse resistance, and episode length.
- Do not retain experimental reward switches, the losing formulation, or training artifacts in the production diff.
- Do not remove or rename existing public configuration fields without a prior deprecation.
- Add no dependencies.
- Do not recreate `source/isaaclab_tasks/test/core/test_pendulum_marl.py` or `test_pendulum_marl_runtime.py`.
- Run `./isaaclab.sh -f` before committing and again before pushing.

---

## File Map

- `source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env.py`
  owns reward state extraction, pure reward computation, and the identical per-agent reward mapping.
- `source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env_cfg.py`
  owns reward coefficients; add an action coefficient only if the cosine candidate wins.
- `source/isaaclab_tasks/test/benchmarking/configs.yaml`
  owns the production training threshold whose scale must match the winning reward.
- `source/isaaclab_tasks/changelog.d/pendulum-marl-refactor.major.rst`
  describes the user-visible shared-reward behavior and checkpoint migration impact.
- `/tmp/isaaclab-pendulum-reward-experiments/`
  stores ignored candidate training JSON; it is never staged.

### Task 1: Establish the Shared Quadratic Candidate

**Files:**
- Modify temporarily: `source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env.py:96-112,239-269`
- Inspect: `source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env_cfg.py:55-64`
- Test: inline assertions only; no test file

**Interfaces:**
- Consumes: normalized policy state already cached by `PendulumMARLEnv`, `self.terminated_dict`, and `self.step_dt`.
- Produces: `compute_quadratic_team_reward` returning `torch.Tensor` with shape `(num_envs,)`, and `_get_rewards() -> dict[str, torch.Tensor]` with equal `cart` and `pendulum` values.

- [ ] **Step 1: Run a failing inline contract probe against the current split reward**

Run:

```bash
./isaaclab.sh -p -c "import torch; from isaaclab_tasks.core.pendulum.pendulum_marl_env import compute_rewards; r = compute_rewards(1.0, -2.0, -0.01, -1.0, -0.01, -1.0, -0.01, torch.tensor([0.2]), torch.tensor([0.1]), torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([0.4]), torch.tensor([False])); assert torch.equal(r['cart'], r['pendulum'])"
```

Expected: `AssertionError`, demonstrating that the current agents receive different rewards for the same transition.

- [ ] **Step 2: Replace the split helper with the temporary quadratic team helper**

Implement this pure function in `pendulum_marl_env.py`:

```python
@torch.jit.script
def compute_quadratic_team_reward(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.square(pole_pos)
    rew_pendulum_pos = rew_scale_pendulum_pos * torch.square(normalize_angle(pole_pos + pendulum_pos))
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)
    rew_pole_vel = rew_scale_pole_vel * torch.abs(pole_vel)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.abs(pendulum_vel)
    return (
        rew_alive
        + rew_termination
        + rew_pole_pos
        + rew_pendulum_pos
        + rew_cart_vel
        + rew_pole_vel
        + rew_pendulum_vel
    ) * step_dt
```

Update `_get_rewards` to call it and map the same tensor to both agents:

```python
team_reward = compute_quadratic_team_reward(
    self.cfg.rew_scale_alive,
    self.cfg.rew_scale_terminated,
    self.cfg.rew_scale_cart_vel,
    self.cfg.rew_scale_pole_pos,
    self.cfg.rew_scale_pole_vel,
    self.cfg.rew_scale_pendulum_pos,
    self.cfg.rew_scale_pendulum_vel,
    self.joint_vel[:, self._cart_dof_idx[0]],
    normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]]),
    self.joint_vel[:, self._pole_dof_idx[0]],
    normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]]),
    self.joint_vel[:, self._pendulum_dof_idx[0]],
    math.prod(self.terminated_dict.values()),
    self.step_dt,
)
return {agent: team_reward for agent in self.cfg.possible_agents}
```

- [ ] **Step 3: Verify the quadratic formula and time scaling inline**

Run:

```bash
./isaaclab.sh -p -c "import torch; from isaaclab_tasks.core.pendulum.pendulum_marl_env import compute_quadratic_team_reward; r = compute_quadratic_team_reward(1.0, -2.0, -0.01, -1.0, -0.01, -1.0, -0.01, torch.tensor([0.2]), torch.tensor([0.1]), torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([0.4]), torch.tensor([False]), 0.5); expected = (1.0 - 0.01 - 0.01 - 0.002 - 0.003 - 0.004) * 0.5; torch.testing.assert_close(r, torch.tensor([expected]))"
```

Expected: exit code 0.

- [ ] **Step 4: Run a one-iteration environment smoke training**

Run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 42 --max_iterations 1 --output_path /tmp/isaaclab-pendulum-reward-experiments/quadratic-smoke
```

Expected: exit code 0, one completed iteration, and no traceback.

### Task 2: Benchmark the Shared Quadratic Candidate

**Files:**
- Read: JSON under `/tmp/isaaclab-pendulum-reward-experiments/quadratic-*`
- Modify: none beyond the temporary Task 1 implementation

**Interfaces:**
- Consumes: active `compute_quadratic_team_reward` implementation from Task 1.
- Produces: three 300-iteration benchmark JSON files and a four-metric summary per seed.

- [ ] **Step 1: Train seeds 42, 43, and 44 sequentially**

Run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 42 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/quadratic-42
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 43 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/quadratic-43
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 44 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/quadratic-44
```

Expected: each run completes 300 iterations with no traceback.

- [ ] **Step 2: Extract comparable metrics from each JSON**

Run:

```bash
jq '{seed:.run.seed, final_50_success:(.learning.success_rate.series_per_iter[-50:]|add/length), max_success:(.learning.success_rate.series_per_iter|max), collapse_from_peak:((.learning.success_rate.series_per_iter|max)-(.learning.success_rate.series_per_iter[-50:]|add/length)), final_50_episode_length:(.learning.ep_length.series_per_iter[-50:]|add/length), final_50_reward:(.learning.reward.series_per_iter[-50:]|add/length), wall_time_s:.runtime.total_wall_time_s}' /tmp/isaaclab-pendulum-reward-experiments/quadratic-42/benchmark_training_*.json /tmp/isaaclab-pendulum-reward-experiments/quadratic-43/benchmark_training_*.json /tmp/isaaclab-pendulum-reward-experiments/quadratic-44/benchmark_training_*.json
```

Expected: numeric values for every field. Record them in the working notes for the final comparison.

### Task 3: Establish and Benchmark the Bounded Cosine Candidate

**Files:**
- Modify temporarily: `source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env.py:96-112,239-269`
- Modify temporarily: `source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env_cfg.py:55-64`
- Test: inline assertions only; no test file

**Interfaces:**
- Consumes: unbounded policy-space cart and pendulum actions before physical scaling, plus the physical link state.
- Produces: `compute_cosine_team_reward` returning `torch.Tensor` with shape `(num_envs,)`, equal per-agent rewards, and three 300-iteration benchmark JSON files.

- [ ] **Step 1: Run the missing-helper probe before implementation**

Run:

```bash
./isaaclab.sh -p -c "from isaaclab_tasks.core.pendulum.pendulum_marl_env import compute_cosine_team_reward"
```

Expected: `ImportError`, because the cosine helper does not exist yet.

- [ ] **Step 2: Add the temporary cosine team helper**

Add `rew_scale_action = -0.01` to `PendulumMARLEnvCfg` and implement:

```python
@torch.jit.script
def compute_cosine_team_reward(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    rew_scale_action: float,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    cart_action: torch.Tensor,
    pendulum_action: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    lower_angle = normalize_angle(pole_pos + pendulum_pos)
    lower_velocity = pole_vel + pendulum_vel
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.cos(pole_pos)
    rew_pendulum_pos = rew_scale_pendulum_pos * torch.cos(lower_angle)
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)
    rew_pole_vel = rew_scale_pole_vel * torch.abs(pole_vel)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.abs(lower_velocity)
    rew_action = rew_scale_action * (
        torch.sum(torch.square(cart_action), dim=1) + torch.sum(torch.square(pendulum_action), dim=1)
    )
    return (
        rew_alive
        + rew_termination
        + rew_pole_pos
        + rew_pendulum_pos
        + rew_cart_vel
        + rew_pole_vel
        + rew_pendulum_vel
        + rew_action
    ) * step_dt
```

Set `rew_scale_pole_pos = 1.0` and `rew_scale_pendulum_pos = 1.0` while this candidate is active. Update `_get_rewards` to pass `self.actions["cart"]`, `self.actions["pendulum"]`, and `self.step_dt`, then return the same tensor for every possible agent.

- [ ] **Step 3: Verify upright preference, world-relative damping, effort, and time scaling**

Run:

```bash
./isaaclab.sh -p -c "import math, torch; from isaaclab_tasks.core.pendulum.pendulum_marl_env import compute_cosine_team_reward; z=torch.zeros(1); a=torch.zeros((1,1)); upright=compute_cosine_team_reward(1.0,-2.0,-0.01,1.0,-0.01,1.0,-0.01,-0.01,z,z,z,z,z,a,a,torch.tensor([False]),0.5); tilted=compute_cosine_team_reward(1.0,-2.0,-0.01,1.0,-0.01,1.0,-0.01,-0.01,z,torch.tensor([math.pi/2]),z,z,z,a,a,torch.tensor([False]),0.5); effort=compute_cosine_team_reward(1.0,-2.0,-0.01,1.0,-0.01,1.0,-0.01,-0.01,z,z,z,z,z,torch.ones((1,1)),torch.ones((1,1)),torch.tensor([False]),0.5); assert upright.item() == 1.5; assert tilted.item() < upright.item(); assert effort.item() < upright.item()"
```

Expected: exit code 0.

- [ ] **Step 4: Run the cosine smoke and three training seeds**

Run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 42 --max_iterations 1 --output_path /tmp/isaaclab-pendulum-reward-experiments/cosine-smoke
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 42 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/cosine-42
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 43 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/cosine-43
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 44 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/cosine-44
```

Expected: all four runs complete without traceback.

- [ ] **Step 5: Extract the cosine metrics**

Run:

```bash
jq '{seed:.run.seed, final_50_success:(.learning.success_rate.series_per_iter[-50:]|add/length), max_success:(.learning.success_rate.series_per_iter|max), collapse_from_peak:((.learning.success_rate.series_per_iter|max)-(.learning.success_rate.series_per_iter[-50:]|add/length)), final_50_episode_length:(.learning.ep_length.series_per_iter[-50:]|add/length), final_50_reward:(.learning.reward.series_per_iter[-50:]|add/length), wall_time_s:.runtime.total_wall_time_s}' /tmp/isaaclab-pendulum-reward-experiments/cosine-42/benchmark_training_*.json /tmp/isaaclab-pendulum-reward-experiments/cosine-43/benchmark_training_*.json /tmp/isaaclab-pendulum-reward-experiments/cosine-44/benchmark_training_*.json
```

Expected: numeric final-50 success, peak collapse, episode length, reward, and duration for every seed.

### Task 4: Select the Winner and Resolve Ambiguous Results

**Files:**
- Read: baseline JSON in `/tmp/isaaclab-pendulum-training-7025*`
- Read: candidate JSON in `/tmp/isaaclab-pendulum-reward-experiments/`
- Modify: none

**Interfaces:**
- Consumes: three-seed metric summaries for baseline, quadratic, and cosine rewards.
- Produces: one selected reward formulation with written selection evidence.

- [ ] **Step 1: Build the comparison table**

For each formulation, calculate:

```text
primary_score = min(seed_42_final_50_success, seed_43_final_50_success, seed_44_final_50_success)
aggregate_score = mean(seed_42_final_50_success, seed_43_final_50_success, seed_44_final_50_success)
worst_collapse = max(seed_42_collapse, seed_43_collapse, seed_44_collapse)
```

Expected: one row per formulation with all three scores and per-seed episode length.

- [ ] **Step 2: Apply the selection rule**

Choose the candidate with the highest `primary_score`. If candidate primary scores differ by no more than `0.05`, use higher `aggregate_score`, then lower `worst_collapse`, then higher mean episode length.

Expected: one provisional winner and an explicit reason based on the ordered criteria.

- [ ] **Step 3: Extend only relevant candidates when required**

If the candidates are within `0.05`, either candidate has a clear upward final-50 trend, or neither candidate has `primary_score >= 0.80`, reactivate each candidate in turn and run all three seeds from scratch:

```bash
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 42 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/quadratic-42-600
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 43 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/quadratic-43-600
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 44 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/quadratic-44-600
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 42 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/cosine-42-600
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 43 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/cosine-43-600
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 44 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/cosine-44-600
```

Expected: use the 600-iteration final-50 metrics for the final decision. If neither candidate reaches the quality bar, preserve the evidence, report the limitation, and pause for user direction before Task 5; do not claim robust training or productionize a candidate merely because it is less poor.

### Task 5: Productionize the Winning Reward

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env_cfg.py`
- Modify: `source/isaaclab_tasks/test/benchmarking/configs.yaml:118-124`
- Modify: `source/isaaclab_tasks/changelog.d/pendulum-marl-refactor.major.rst`
- Test: inline assertions and runtime benchmark commands only

**Interfaces:**
- Consumes: the selected temporary helper and its three-seed evidence.
- Produces: one `compute_rewards` helper returning `torch.Tensor` and identical rewards for every configured agent.

- [ ] **Step 1: Delete the losing experimental helper and rename the winner**

Retain only the winning formula as `compute_rewards` and keep its explicit `torch.Tensor` return annotation. If quadratic wins, replace the `compute_quadratic_team_reward` identifier in both its definition and `_get_rewards` call with `compute_rewards`, without changing the argument list. If cosine wins, perform the same identifier-only replacement for `compute_cosine_team_reward`. In both cases, preserve this exact return statement:

```python
return {agent: team_reward for agent in self.cfg.possible_agents}
```

Expected: no reward-mode flag, alternate helper, environment-variable selector, or hard-coded duplicate per-agent formula remains.

- [ ] **Step 2: Normalize the production configuration**

If cosine wins, retain:

```python
rew_scale_pole_pos = 1.0
rew_scale_pendulum_pos = 1.0
rew_scale_action = -0.01
```

If quadratic wins, retain the existing negative position coefficients and remove the temporary action coefficient. In either case, do not rename existing configuration fields.

- [ ] **Step 3: Update the benchmark reward threshold deterministically**

Compute the skrl top-eight reward mean for each winning 300-iteration seed, matching `_extract_reward` in `env_benchmark_test_utils.py`. For a cosine winner, run:

```bash
jq -s '[.[] | (.learning.reward.series_per_iter | sort | .[-8:] | add / length)] | min * 0.8 * 10 | floor / 10' /tmp/isaaclab-pendulum-reward-experiments/cosine-42/benchmark_training_*.json /tmp/isaaclab-pendulum-reward-experiments/cosine-43/benchmark_training_*.json /tmp/isaaclab-pendulum-reward-experiments/cosine-44/benchmark_training_*.json
```

For a quadratic winner, run the same expression against the three `quadratic-42`, `quadratic-43`, and `quadratic-44` paths. Set `lower_thresholds.reward` to the printed one-decimal value. Keep `episode_length: 150`, `max_iterations: 300`, and `duration: 500` unchanged unless the final training evidence directly violates them.

- [ ] **Step 4: Add the behavior change to the existing changelog fragment**

Add under `Changed`:

```rst
* Changed ``Isaac-Pendulum-MARL-Direct`` to give both agents a shared team
  reward aligned with upright balancing. Retrain policies created with the
  previous split per-agent rewards.
```

- [ ] **Step 5: Run the pure reward contract probe**

Run the applicable Task 1 or Task 3 inline formula probe after replacing the temporary helper name with `compute_rewards`.

Expected: exit code 0.

- [ ] **Step 6: Verify registry and configuration loading without adding a test file**

Run:

```bash
./isaaclab.sh -p -c "import gymnasium as gym; import isaaclab_tasks; spec=gym.spec('Isaac-Pendulum-MARL-Direct'); assert spec.entry_point == 'isaaclab_tasks.core.pendulum:PendulumMARLEnv'; assert spec.kwargs['env_cfg_entry_point'].endswith('PendulumMARLEnvCfg')"
```

Expected: exit code 0.

- [ ] **Step 7: Run final production training for all three seeds**

Run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 42 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/final-42
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 43 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/final-43
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 44 --max_iterations 300 --output_path /tmp/isaaclab-pendulum-reward-experiments/final-44
```

Expected: final production code reproduces the selected candidate's robust-success behavior without traceback.

If Task 4 required 600 iterations, also run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 42 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/final-42-600
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 43 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/final-43-600
env OMNI_KIT_ACCEPT_EULA=YES uv run --frozen --extra isaacsim --extra skrl isaaclab benchmark training --rl_library skrl --task Isaac-Pendulum-MARL-Direct --seed 44 --max_iterations 600 --output_path /tmp/isaaclab-pendulum-reward-experiments/final-44-600
```

Expected: the final code also reproduces the extended-horizon selection evidence.

- [ ] **Step 8: Run focused static checks**

Run:

```bash
git diff --check
rg -n "quadratic_team|cosine_team|reward_mode|PENDULUM_REWARD" source/isaaclab_tasks/isaaclab_tasks/core/pendulum
git status --short
```

Expected: clean diff check; no temporary selector names; only intended source, benchmark, changelog, planning files, and ignored `env_isaaclab` appear.

- [ ] **Step 9: Run full pre-commit twice if formatting changes files**

Run `./isaaclab.sh -f`. If it modifies files, review only those diffs, stage intended formatter edits, restore unrelated edits, and run `./isaaclab.sh -f` again.

Expected: final run passes every hook.

- [ ] **Step 10: Commit the production reward**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env.py source/isaaclab_tasks/isaaclab_tasks/core/pendulum/pendulum_marl_env_cfg.py source/isaaclab_tasks/test/benchmarking/configs.yaml source/isaaclab_tasks/changelog.d/pendulum-marl-refactor.major.rst
git commit -m "Improve pendulum MARL team reward" -m "Give both policies one success-aligned cooperative objective and update the training benchmark for its new return scale."
```

Expected: one focused production commit; `env_isaaclab` remains untracked.

### Task 6: Review and Update the Draft PR

**Files:**
- Review: complete branch diff against its merge base
- Modify: draft PR 7025 description through GitHub API

**Interfaces:**
- Consumes: committed winning reward, training evidence, and clean verification output.
- Produces: reviewed, pushed branch and an accurate draft PR description.

- [ ] **Step 1: Perform the requested self-review**

Review `git diff $(git merge-base HEAD origin/main)..HEAD` for correctness, public API compatibility, reward/success alignment, stale temporary code, configuration consistency, and unrelated changes. Run the `pr-review-toolkit` workflow and record findings before changing code.

Expected: either no actionable findings or focused follow-up commits for every finding.

- [ ] **Step 2: Re-run verification after review fixes**

Run the pure reward probe, registry probe, `git diff --check`, and `./isaaclab.sh -f` after any review change.

Expected: all checks pass on the exact commit to be pushed.

- [ ] **Step 3: Push only to the fork remote**

```bash
git push antoine antoiner/pendulum-marl-refactor
```

Expected: branch updates on `antoine`; never push to `origin`.

- [ ] **Step 4: Update the draft PR description**

Use `gh api` to update PR 7025 with the winning reward formula, unchanged success definition, per-seed final-50 success table, any 600-iteration extension, focused verification, and the full pre-commit result.

Expected: PR remains a draft and accurately states both successful evidence and any remaining training instability.

- [ ] **Step 5: Report the outcome**

Report the selected reward, exact seed results, whether the 80% worst-seed quality bar was met, commit hashes, verification commands, and the draft PR link.

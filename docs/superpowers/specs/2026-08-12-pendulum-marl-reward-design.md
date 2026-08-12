# Pendulum MARL Reward Experiment

## Goal

Improve the robustness of the `Isaac-Pendulum-MARL-Direct` policy without
changing the task's success definition. Both agents will receive the same team
reward so that their optimization target matches the cooperative task.

Success remains an episode-level metric: both physical links must stay within
15 degrees of upright for the final 1 second of an episode that reaches its time
limit without an early failure.

## Reward Candidates

Both candidates retain the current actions, observations, reset distribution,
termination conditions, policy configuration, and success metric. Both return
the same reward tensor for the `cart` and `pendulum` agents and scale reward by
the environment control-step duration.

Define the physical link state as:

- `upper_angle = wrap(pole_joint_angle)`
- `lower_angle = wrap(pole_joint_angle + pendulum_joint_angle)`
- `upper_velocity = pole_joint_velocity`
- `lower_velocity = pole_joint_velocity + pendulum_joint_velocity`

### Candidate 1: bounded cosine team reward

Use the four-bar cartpole's bounded upright objective and damping structure:

```text
reward = (
    alive
    + cos(upper_angle)
    + cos(lower_angle)
    - 0.01 * abs(cart_velocity)
    - 0.01 * abs(upper_velocity)
    - 0.01 * abs(lower_velocity)
    - 0.01 * (cart_action**2 + pendulum_action**2)
    + termination_penalty
) * step_dt
```

The alive and termination terms retain their current weights of `1.0` and
`-2.0`. Actions are the normalized policy actions, so the effort term treats
both agents consistently despite their different physical action scales.

This candidate directly rewards both physical links being upright, bounds the
angle contribution, damps physical link motion, and discourages unnecessarily
aggressive control.

### Candidate 2: shared quadratic team reward

Use the current quadratic shaping terms, but sum them into one team reward:

```text
reward = (
    alive
    - upper_angle**2
    - lower_angle**2
    - 0.01 * abs(cart_velocity)
    - 0.01 * abs(pole_joint_velocity)
    - 0.01 * abs(pendulum_joint_velocity)
    + termination_penalty
) * step_dt
```

The alive and termination terms retain their current weights. This candidate
isolates the effect of correcting multi-agent credit assignment while otherwise
preserving the existing reward objective. It intentionally has no new action
penalty.

## Experiment

Evaluate the current split-reward baseline and both candidates with the same
training setup:

- PhysX backend
- skrl PPO configuration already registered for the task
- 4,096 environments
- seeds 42, 43, and 44
- 300 training iterations per candidate and seed

Runs are sequential on the same GPU. Temporary experimental selection code and
artifacts are not retained in the production diff.

For every run, record:

- mean success rate over the final 50 iterations
- maximum success rate
- drop from maximum success to final-50 mean
- final-50 episode length
- wall-clock training duration

Raw return is recorded for diagnosis but is not used to compare candidates,
because the reward formulations have different numeric scales.

## Selection

Rank candidates primarily by the lowest final-50 success rate across the three
seeds. This makes resistance to the seed-44-style collapse more important than
a high average supported by only easy seeds. Break close results, within five
percentage points, by:

1. higher mean final-50 success across seeds;
2. smaller worst-seed drop from peak success;
3. longer final-50 episode length.

The preferred quality bar is at least 80% final-50 success on every seed. If the
leading candidates are within five percentage points, are still trending at
iteration 300, or neither reaches the quality bar, extend the relevant runs to
600 iterations before choosing. If neither candidate trains robustly after the
extension, report that result rather than treating the less-bad reward as a
successful solution.

## Production Change and Verification

Keep only the winning reward formulation. Do not retain a runtime reward-mode
switch or add pendulum-specific test files. Update configuration names and
benchmark expectations only where required by the winning formulation, while
following the repository's public-API deprecation policy.

Verification consists of:

- focused environment startup and rollout smoke checks;
- the winning three-seed training evidence;
- registry and configuration loading checks;
- repository pre-commit checks;
- review of the final diff for unrelated generated or formatter changes.

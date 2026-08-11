# Hand-off: skrl failures on the classic tasks

## Summary

A full Cartpole + Ant + Humanoid benchmark sweep produced **201 failures out of 474 rows.
Every failure is `skrl`.** No other RL library failed a single row.

There are **two independent issues**, not one.

| # | Scope | Rows | Status |
|---|---|---|---|
| A | skrl + non-box observation/action spaces | 192 | root cause confirmed — fix location identified |
| B | skrl + Humanoid-AMP | 9 | root cause confirmed — fix location identified |

Both share the same underlying gap: **Odin dispatch doesn't know how to select
the right `--agent` or `--algorithm` for tasks with non-default skrl configs.**

---

## Environment

- Branch `antoiner/feat/odin-v2`, develop as of 2026-08-11
- Image `nvcr.io/nvidian/antoiner-isaac-lab@sha256:beeca2331adaacdf4b25a8d0e316298e5439f0d1f7c85f4f39614896c880609c`
- ovphysx 0.5.10 + ovstage 0.1.1.355824 (includes PR #7021), Newton 1.5.0rc2, warp 1.16.0
- Dispatch `20260811-124406`, OSMO pool `isaac-lab-test-l40-07` (**L40**), 3 seeds (42/43/44)
- Local state: `odin_runs/20260811-124406/dispatch.json`

## Control: nothing else is broken

| library | result |
|---|---|
| rl_games | 66/66 pass |
| rsl_rl | 66/66 pass |
| sb3 | 45/45 pass |
| skrl on core tasks (Ant, Cartpole, Humanoid — manager **and** Direct) | 66/66 pass |

skrl itself is fine. Only specific task/preset combinations fail.

---

## Issue A — non-box spaces (192 rows)

`IsaacContrib-Cartpole-Showcase-Direct` (168 failed / 24 passed) and
`IsaacContrib-Cartpole-Camera-Showcase-Direct` (24 failed / 6 passed).

Presets are named `<observation-space>_<action-space>`. The split is total:

| preset | skrl |
|---|---|
| `(none)`, `box_box` | **pass** |
| `box_discrete`, `box_multidiscrete` | fail |
| `dict_box`, `dict_discrete`, `dict_multidiscrete` | fail |
| `tuple_box`, `tuple_discrete`, `tuple_multidiscrete` | fail |
| `discrete_box`, `discrete_discrete`, `discrete_multidiscrete` | fail |
| `multidiscrete_box`, `multidiscrete_discrete`, `multidiscrete_multidiscrete` | fail |

Error:

```
Error: The size of tensor a (12288) must match the size of tensor b (4096)
       at non-singleton dimension 0
```

`4096` is `num_envs`; `12288 = 3 x 4096`. Consistent with a continuous Gaussian
policy outputting `(num_envs, num_categories)` actions into an env that expects a
scalar index per env for `Discrete(3)`.

**Physics-independent**: exactly 42 failures under each of `isaacsim_physx`,
`newton_kamino`, `newton_mjwarp`, `ovphysx`. This is not a simulation bug.

### Root cause (confirmed)

The Odin dispatch does not wire the preset to the matching skrl agent config.

`discover.py` → `expand_rows()` emits rows like:

```json
{"task_id": "IsaacContrib-Cartpole-Showcase-Direct", "rl_library": "skrl", "presets": ["box_discrete"]}
```

The dispatch template calls:

```bash
isaaclab benchmark training --rl_library skrl --task IsaacContrib-Cartpole-Showcase-Direct \
  ... presets=box_discrete
```

No `--agent` flag is passed. In `benchmark_train_skrl.py` (line 263–268):

```python
# tools/odin/templates/dispatch.yaml.j2 never injects --agent
if args_cli.agent is None:
    agent_cfg_entry_point = "skrl_cfg_entry_point"   # always resolves to box_box yaml
```

`skrl_cfg_entry_point` → `skrl_box_box_ppo_cfg.yaml` → **GaussianMixin** (continuous policy,
outputs `(num_envs, 3)` action tensor). The env was configured with `box_discrete` preset
→ `Discrete(3)` action space, which expects a scalar index per env. Shape mismatch →
crash in `CartpoleShowcaseEnv._apply_action()`.

The task kwargs *do* declare `agent_preset_compatibility` mapping every preset to its
correct entry point (e.g., `skrl_box_discrete_cfg_entry_point` →
`skrl_box_discrete_ppo_cfg.yaml` with `CategoricalMixin`), but `discover.py` never
reads or emits this information.

### Fix location

**`tools/odin/discover.py`** — `discover_tasks()` / `expand_rows()`.

When a task's kwargs contain `agent_preset_compatibility`, resolve the matching
agent entry point for each (library, preset) pair and propagate it into the dispatch
row. Two concrete options:

1. Add `agent_overrides: dict[str, str]` to `DiscoveredTask.Mode` (keyed by rl_library,
   value is the matching entry point) and add `--agent <ep>` to `PlannedRow.overrides`
   in `expand_rows()`.
2. Add a top-level `agent` field to `PlannedRow` and render it in the dispatch
   template as `--agent {{ row.agent }}`.

Both require touching `plan.py`, `discover.py`, and `dispatch.yaml.j2`.
The dispatch template already appends `row.overrides` verbatim, so option 1 is
the smallest diff.

Also check `source/isaaclab/isaaclab/benchmark/entrypoints/backends/skrl/benchmark_train_skrl.py`
line 263–268 — the logic there is correct once `--agent` is passed; no changes needed there.

### Local reproducer

```bash
# Failing (box_discrete preset → Discrete(3) action space, but loads box_box GaussianMixin)
isaaclab benchmark training --rl_library skrl \
  --task IsaacContrib-Cartpole-Showcase-Direct \
  --num_envs 4096 --max_iterations 5 \
  --output_path /tmp/test_a presets=box_discrete

# Passing control (correct agent config auto-selected for default box_box preset)
isaaclab benchmark training --rl_library skrl \
  --task IsaacContrib-Cartpole-Showcase-Direct \
  --num_envs 4096 --max_iterations 5 \
  --output_path /tmp/test_a presets=box_box

# Passing workaround (manually specify the right agent)
isaaclab benchmark training --rl_library skrl \
  --task IsaacContrib-Cartpole-Showcase-Direct \
  --num_envs 4096 --max_iterations 5 \
  --output_path /tmp/test_a \
  --agent skrl_box_discrete_cfg_entry_point presets=box_discrete
```

---

## Issue B — Humanoid-AMP (9 rows)

`IsaacContrib-Humanoid-AMP-Dance-Direct`, `-Run-Direct`, `-Walk-Direct` — all 3/3 fail.

Distinct from Issue A: no space presets involved (`physics: None`, `presets: []`).
AMP is skrl-only, so there is **no cross-library control** for these tasks.

### Root cause (confirmed)

Same dispatch gap as Issue A, but for the algorithm rather than the preset.

AMP tasks register only `skrl_amp_cfg_entry_point` (not `skrl_cfg_entry_point`).
The dispatch template never passes `--algorithm AMP`. In `benchmark_train_skrl.py`:

```python
# --algorithm defaults to "PPO"
agent_cfg_entry_point = "skrl_cfg_entry_point"   # derived from algorithm=="ppo"
```

`resolve_task_config("IsaacContrib-Humanoid-AMP-Walk-Direct", "skrl_cfg_entry_point")`
fails because the task only has `skrl_amp_cfg_entry_point` in its kwargs.

Note: the motion reference files (`humanoid_walk.npz` etc.) are bundled in
`source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp/motions/` — a missing
motion file is **not** the cause.

### Fix location

Same as Issue A — `tools/odin/discover.py`. When the only skrl entry point for a
task is `skrl_amp_cfg_entry_point` (or any non-`skrl_cfg_entry_point` key), the
discover step must inject `--algorithm AMP` (or `--agent skrl_amp_cfg_entry_point`)
into the dispatch row.

A reliable heuristic: if a task declares `skrl_amp_cfg_entry_point` but not
`skrl_cfg_entry_point`, add `--agent skrl_amp_cfg_entry_point` to the row's
overrides. This composes naturally with the Issue A fix.

### Local reproducer

```bash
# Failing — defaults to --algorithm PPO → tries skrl_cfg_entry_point → not registered
isaaclab benchmark training --rl_library skrl \
  --task IsaacContrib-Humanoid-AMP-Walk-Direct \
  --num_envs 4096 --max_iterations 5 \
  --output_path /tmp/test_b

# Passing workaround — explicitly select the AMP algorithm or agent
isaaclab benchmark training --rl_library skrl \
  --task IsaacContrib-Humanoid-AMP-Walk-Direct \
  --num_envs 4096 --max_iterations 5 \
  --output_path /tmp/test_b \
  --algorithm AMP
```

---

## The unified fix

Both issues reduce to one gap: `discover.py` never propagates how to select a
non-default skrl agent config to the dispatch row.

Suggested approach:

1. In `discover_tasks()`, read `agent_preset_compatibility` from each task's gym
   kwargs.  Build a reverse mapping `{(rl_library, preset_or_none) → entry_point}`.
2. Store it in `DiscoveredTask` (or thread it into `_legal_modes`).
3. In `expand_rows()`, for each (library, mode) row where a specific entry point is
   known, emit `--agent <entry_point>` as an override string.
4. `PlannedRow.overrides` already forwards verbatim tokens to the command line;
   no template change is needed for option 1 above.

Files touched: `tools/odin/discover.py`, `tools/odin/plan.py` (if you add a
typed `agent` field rather than using overrides).

---

## Caveat that affects any per-library comparison

**skrl has no `check_nan`.** In the sibling locomotion dispatch (`20260811-114225`),
every `newton_mjwarp` rough-terrain row that skrl reported as *completed* had in fact
diverged to NaN — rsl_rl and sb3 aborted on the same configurations. So across these
two runs skrl simultaneously **over-reports** failures (here) and **under-reports**
them (there). Do not read "skrl completed" as "skrl trained successfully" on mjwarp.

Adding a NaN guard to the skrl training backend is a separate, worthwhile task.

---

## Reproducing the breakdown

```bash
cd /home/antoiner/Documents/IsaacLab
python3 - <<'EOF'
import json,collections
j=json.load(open('odin_runs/20260811-124406/dispatch.json'))['jobs']
agg=collections.defaultdict(lambda: collections.Counter())
for x in j:
    if x['rl_library']=='skrl':
        key=(x['task_id'], ",".join(x.get('presets') or []) or "(none)")
        agg[key][x['status']]+=1
for k in sorted(agg): print(f"{k[0]:46s} {k[1]:30s} {dict(agg[k])}")
EOF
```

<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Franka suction parcel sorting

`IsaacContrib-Conveyor-Suction-Sorting` demonstrates replicated force-based Newton suction
and shared success-guided reset sampling. Twelve differently sized parcels circulate through
a compact workcell with a gravity-drop feeder. A Franka places parcels onto five size-coded
outlets, whose powered belts deliver them into physical bins. The policy controls the arm
and vacuum; belts run continuously.

![Franka suction parcel sorting](../../../../../docs/source/_static/environment-previews/conveyor_suction_sorting.jpg)

The tool has four visible suction cups of 12 mm radius on the existing Panda hand. Suction requires a nearby,
aligned parcel face and applies the upstream Newton prototype's compliant, capped forces.
It does not move parcels by writing their poses. The original fingers are visually hidden
and their collisions disabled. Existing finger-gripper policies require retraining.

## Experimental dependency

Newton's [surface-gripper prototype](https://github.com/newton-physics/newton/pull/3974)
is not merged. This branch pins an experimental public extraction of that implementation
and its matching MuJoCo dependencies in `pyproject.toml` and `uv.lock`; it does not vendor
the seal implementation into Isaac Lab. Stock Newton releases do not provide this API yet.

## Training and playback

The default configuration uses 512 environments, 6,000 PPO updates, and 240-second episodes.
Each verified delivery counts as a success, and the robot continues with another parcel in the
same scene. Reliable learning from scratch has not yet been validated.

```bash
uv run isaaclab train --task IsaacContrib-Conveyor-Suction-Sorting \
  --rl_library rsl_rl --num_envs 512 --device cuda:0

uv run isaaclab play --task IsaacContrib-Conveyor-Suction-Sorting \
  --rl_library rsl_rl --checkpoint /path/to/model.pt \
  --num_envs 1 --device cuda:0 --viz newton_gl --real-time
```

With Isaac Sim installed, use `--viz kit` for RTX visualization.

The policy outputs three Cartesian target increments and one binary vacuum command. Standard
differential IK maintains a downward tool orientation. State observations include the
selected parcel's pose, velocity and dimensions, robot and tool states, and suction state.
The 52 observations include whether the selected parcel has completed its required lift;
this history remains observable after lowering or releasing it. Earlier 51-input checkpoints
require their matching configuration.
Parcel dimensions vary across twelve fixed cuboid assets spanning five size classes.
Parcels are 8.5–19 cm long and weigh approximately 64–431 g. The receiving lanes have
24 cm openings to accommodate the largest packages. Earlier suction checkpoints require
their matching parcel catalogue and tool configuration; the observation shape alone does not imply compatibility.
The correct destination is determined by package dimensions; a destination command or
size-class label is not supplied to the actor. Colors make the scene readable without
becoming policy inputs.

## Reset distribution and success

The public `isaaclab_tasks.utils.ResetStateSampler` tracks recent outcomes and chooses
reset rows. Task-local generators produce belt, approach/grasp, carry and near-release states;
ordinary scene/asset writers restore them. Other parcels are shuffled among the remaining
belt and feeder slots to vary the order of subsequent pickups. The same sampler is used by the original
conveyor curriculum and can select procedurally generated or cached reset states.
It does not add another environment manager or a second scene-state serializer.

All twelve parcels begin unsorted in every reset phase. A verified physical receipt earns
a reward of +10 and selects the next reachable unsorted parcel without resetting the world.
The first receipt after a sampled reset immediately records a successful outcome for that
reset row. Later receipts still earn rewards and increment delivery counters, but are not
attributed to the earlier reset state. A world reset before that first receipt records a
failed row outcome; a later failure does not overwrite or duplicate an already recorded success.

Training starts receive more practice when the policy has intermediate success, while
every reset row remains eligible. Sampling history is local to each worker and restarts
when the environment is recreated. Training success from an easier reset is not an
end-to-end pickup result. Evaluate separately from full belt starts and report correct-bin
deliveries by size class, wrong-bin deliveries, drops, and complete twelve-parcel batches.

Playback samples full belt starts uniformly. `full_start_success_rate` measures whether a
full home start produces a first receipt; `full_batch_success_rate` separately measures all
twelve physical receipts from that start. A single delivery is a success without requiring
a complete batch. Receiving all twelve parcels ends the episode to replenish the scene;
timeout or losing an unsorted parcel also ends it.

Development results used staged warm-started checkpoints. Resume a compatible checkpoint with
`--checkpoint /path/to/model.pt`; optimizer state is retained by default.

## Facility references

The induction-loop/offload layout follows ordinary distribution-center organization,
including [Dematic's crossbelt sorter layouts](https://www.dematic.com/content/dam/dematic/downloads/brochures/NA_BR_1056_Crossbelt_Sorter.pdf).
The scene uses native primitives with physical decks, guards and bins; decorative
supports remain visual only.

Support owner: Maximilian Krause. This example incubates the Newton suction integration;
the adapter should follow the public upstream API once it stabilizes, and the task can be
relocated if this capability is covered by a smaller maintained example.

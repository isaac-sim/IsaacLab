<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Franka suction parcel sorting

`IsaacContrib-Conveyor-Suction-Sorting` demonstrates replicated force-based Newton suction
and shared success-guided reset sampling. Twelve differently sized parcels circulate through
a compact workcell with a gravity-drop feeder. A Franka places parcels at commanded positions
on five receiving belts, which deliver them into physical bins. The policy controls the arm
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
differential IK maintains a downward tool orientation, with bounded target lead and normal
Franka joint velocity limits. Observations include the selected parcel's pose, velocity and
dimensions, robot and tool states, pickup and placement qualification, and the commanded
parcel-center goal in the robot frame. A small overhead ray scan observes collision geometry
as robot-frame points and hit-validity bits. It uses Newton physics rather than an RTX camera;
the 20 cm ray spacing can miss thin obstacles, and occluded surfaces are not observed.
Earlier checkpoints require retraining with
the new observations and controlled-placement objective.
Parcel dimensions vary across twelve fixed cuboid assets spanning five size classes.
Parcels are 8.5–19 cm long and weigh approximately 64–431 g. The receiving lanes have
24 cm openings to accommodate the largest packages. Earlier suction checkpoints require
their matching parcel catalogue and tool configuration; the observation shape alone does not imply compatibility.
Training samples the destination independently of package dimensions, with small placement
offsets within each supported receiving area. Playback routes each size class to its original
outlet. Both modes supply the goal explicitly, so dimensions cannot replace following the
command. Colors make the scene readable without becoming policy inputs. Goals are restricted
to these reachable, supported receiving areas; this example does not establish transfer to
arbitrary facilities or obstacles.

## Reset distribution and success

The public `isaaclab_tasks.utils.ResetStateSampler` tracks recent outcomes and chooses
reset rows. Task-local generators produce belt, approach/grasp, carry and near-release states
for each parcel and destination. Near-release states use the same goal given to the actor;
ordinary scene/asset writers restore them. Other parcels are shuffled among the remaining
belt and feeder slots to vary the order of subsequent pickups. The same sampler is used by the original
conveyor curriculum and can select procedurally generated or cached reset states.
It does not add another environment manager or a second scene-state serializer.

All twelve parcels begin unsorted in every reset phase. Release is checked before detachment:
the parcel must be upright, close to its commanded position, at most 2 cm above the receiving
surface, and moving slowly relative to either the stationary world or the moving belt.
Excessive vertical or angular velocity, an uncontrolled release, or nongripper arm contact fails
the attempt. Only a qualified placement followed by a settled receipt in the commanded bin
earns credit; a thrown parcel cannot qualify merely by landing in that bin.
A verified physical receipt earns
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

Evaluate the retrained policy from full starts, with held-out reset seeds and destination
commands. Report controlled-placement success and arm contacts alongside physical receipts;
the earlier receipt-only policies are not evidence for this stricter objective.

## Facility references

The induction-loop/offload layout follows ordinary distribution-center organization,
including [Vanderlande's robotic induction at DHL](https://www.vanderlande.com/news-insights/vanderlande-delivers-robot-technology-to-dhl/).
The scene uses native primitives with physical decks, guards and bins; decorative
supports remain visual only.

Support owner: Maximilian Krause. This example incubates the Newton suction integration;
the adapter should follow the public upstream API once it stabilizes, and the task can be
relocated if this capability is covered by a smaller maintained example.

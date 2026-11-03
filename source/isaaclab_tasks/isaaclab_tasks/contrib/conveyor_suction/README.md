<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Franka suction parcel sorting

`IsaacContrib-Conveyor-Suction-Sorting` demonstrates replicated force-based Newton suction
and shared success-guided reset sampling. Twelve differently sized parcels circulate through
a compact workcell with a gravity-drop feeder. A Franka places parcels at commanded positions
on five or six receiving belts, which deliver them into physical bins. The policy controls the arm
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

The default configuration uses 512 environments, 6,000 PPO updates, and 30-second training episodes.
Training practices one assignment per episode: its first verified receipt ends the episode successfully,
and missing the assigned pickup fails the attempt. Playback uses the original five-lane fixture and
240-second episodes, continuing to another parcel after each controlled placement. Each verified
delivery counts as a success. Reliable learning from scratch has not yet been validated.

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
Franka joint velocity limits. The policy bounds Cartesian means to the accepted action range and
starts near rest with a 98% vacuum-hold exploration prior. Either vacuum command retains at least 0.1%
sampling probability during training. Observations include the selected parcel's pose, velocity and
dimensions, robot and tool states, pickup and placement qualification, and the commanded
parcel-center goal in the robot frame. A small overhead ray scan observes collision geometry
as robot-frame points and hit-validity bits. It uses Newton physics rather than an RTX camera;
the 24 cm ray spacing can miss thin obstacles, and occluded surfaces are not observed.
Earlier checkpoints require retraining with
the new observations and controlled-placement objective.
Parcel dimensions vary across twelve fixed cuboid assets spanning five size classes.
Parcels are 8.5–19 cm long and weigh approximately 64–431 g. The receiving lanes have
24 cm openings to accommodate the largest packages. Earlier suction checkpoints require
their matching parcel catalogue and tool configuration; the observation shape alone does not imply compatibility.
Training distributes parallel environments across sixteen coherent physical fixtures: eight have
five receiving lanes and eight have six. Lane angles, radial positions, and heights vary together
with the corresponding goal, bin, and reset geometry. Each environment keeps its fixture for its
lifetime; placement offsets and starting states vary at reset. The original five-lane fixture and
its six-lane extension remain in the catalog. Training samples every active destination independently
of package dimensions, with placement offsets inside its supported receiving area. Playback routes each size class to its original
outlet. Both modes supply the goal explicitly, so dimensions cannot replace following the
command. Colors make the scene readable without becoming policy inputs. Goals are restricted
to these reachable, supported receiving areas; this example does not establish transfer to
arbitrary facilities or obstacles. The interface has 312 observations and four actions. The wider collision scan
changes the spatial meaning of its samples, so earlier checkpoints require their original observation configuration.
New training uses fresh weights. Evaluate separately on held-out fixture layouts before claiming generalization.

## Reset distribution and success

The public `isaaclab_tasks.utils.ResetStateSampler` tracks 49 shared progress buckets:
sixteen points each for controlled descent, held lift/transport, and unheld approach, plus a full-home start.
Each draw independently chooses a parcel, destination, and pickup anchor on the world's physical fixture.
The cached geometry contains 135,168 physical states, while the outcome history stays small enough to
receive repeated attempts. Layout and package variations share progress history instead of fragmenting it.
The Beta sampler favors intermediate success and retains a uniform strategy with weight 0.005.

Connected paths cover contact, lift, transport around the pedestal, and controlled descent.
Earlier approach states rewind the moving parcel around the incoming turn. Held starts restore native
suction seals against the reset geometry before the first observation. Other parcels are shuffled among
the remaining belt and feeder slots. Initially, about 6% of sampled starts are already near enough to
practice a controlled release. These bootstrap states connect continuously to larger descents;
progress levels 9–15 retain at least 6 cm of geometric clearance above the receiving surface.
Released outfeed starts are excluded: the conveyor can complete those deliveries without a policy
action. No reset seeds a receipt or successful outcome.

No parcel starts with delivery credit in any reset phase. Release is checked before detachment:
the parcel must be upright, close to its commanded position, at most 2 cm above the receiving
surface, and moving slowly relative to either the stationary world or the moving belt.
Excessive vertical or angular velocity, an uncontrolled release, or nongripper arm contact fails
the attempt. Only a qualified placement followed by a settled receipt in the commanded bin
earns credit; a thrown parcel cannot qualify merely by landing in that bin.
A verified physical receipt earns a reward of +10. Training retains its original target until that
receipt or a failure; it cannot replace a missed parcel with an easier incoming one.
During continuing playback, a controlled release that has detached and cleared the suction pads
allows the dispatcher to select another parcel before the previous bin arrival. Released parcels retain
their own destinations and receipt checks while the arm handles the next target. Simultaneous receipts
each earn credit; a receipt never changes the current grasp. Playback also replaces missed pickups
with incoming targets. Each handoff rebases the Cartesian controller at the measured tool pose without moving the scene.
The original reset parcel's receipt immediately records a successful outcome for its
shared progress bucket. Other receipts still earn rewards and increment delivery counters, but are not
attributed to the earlier reset state. A world reset before the original parcel's receipt records a
failed row outcome; a later failure does not overwrite or duplicate an already recorded success.

The reward remains sparse: a complete qualified receipt earns +10; failures and motion
regularization retain their existing penalties. There are no reaching, lifting or goal-distance
rewards. Reset-family metrics separately expose placement, carry and acquisition.
`release_reset_success_rate` tracks the three easiest descent levels; `descent_success_rate`
tracks levels 9–15, which start at least 6 cm above the receiving surface. Release-ready bootstrap
success is separate from learning a substantial descent or a full pickup.
Training starts receive more practice when the policy has intermediate success, while
every progress bucket remains eligible. Sampling history is local to each worker and restarts
when the environment is recreated. Training success from an easier reset is not an
end-to-end pickup result. Evaluate separately from full belt starts and report correct-bin
deliveries by size class, wrong-bin deliveries, drops, and complete twelve-parcel batches.

Playback samples full belt starts uniformly. `full_start_success_rate` measures receipt of the
original assigned parcel from a full home start; `full_batch_success_rate` separately measures all
twelve physical receipts from that start. A single training delivery ends its episode successfully.
Playback replenishes the scene after all twelve parcels arrive; timeout or losing an unsorted parcel
also ends the episode.

Evaluate the retrained policy from full starts, with held-out reset seeds and destination
commands. Report controlled-placement success and arm contacts alongside physical receipts;
the earlier receipt-only policies are not evidence for this stricter objective.

## Facility references

The induction-loop/offload layout follows ordinary distribution-center organization,
including [Vanderlande's robotic induction at DHL](https://awlautomation.com/customer-experience-dhl-express).
The scene uses native primitives with physical decks, guards and bins; decorative
supports remain visual only.

Support owner: Maximilian Krause. This example incubates the Newton suction integration;
the adapter should follow the public upstream API once it stabilizes, and the task can be
relocated if this capability is covered by a smaller maintained example.

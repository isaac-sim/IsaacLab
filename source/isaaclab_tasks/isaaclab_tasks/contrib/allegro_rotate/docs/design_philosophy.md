# Allegro Rotate Design Philosophy

This task is an Allegro adaptation of the in-hand rotation process:

```text
good ready pose -> stable grasp cache -> rotate policy from cache
```

The main lesson is that in-hand rotation is not learned reliably from an open-hand reset. The policy needs to start from a stable, physically meaningful grasp state, then learn how to maintain and roll the object.

## Ready Pose First

The initial Allegro pose is designed as a finger cage:

```text
thumb opposes the object,
index/middle/ring provide side support,
the object starts between fingers,
the cylinder axis starts upright.
```

The hand should not fully clamp the cylinder. A tight clamp makes rolling hard. A loose pose makes the object drop. The useful pose is in between: enough contact to hold the cylinder, enough freedom for rolling.

The final tuned pose is copied directly into cfg constants:

```text
HAND_ROOT_POS
HAND_ROOT_ROT
CYLINDER_INIT_POS
CYLINDER_INIT_ROT
ALLEGRO_READY_JOINT_POS
```

The submitted task does not depend on a pose-authoring USD at reset time.

## Cache Before Policy

The grasp cache is not a policy and does not store trajectories. It stores stable reset states:

```text
hand joint positions,
object position,
object rotation.
```

During cache generation, zero action means "hold the current joint target" under the position controller. This filters states where the hand can actually hold the cylinder before RL starts.

The rotate policy then samples from this cache. This is the same high-level pattern that made robust.

## Reward Choice

The reward intentionally follows the simpler rotation-focused rotate reward:

```text
reward rotation about the target axis,
penalize object linear motion,
penalize large hand deviation from the reset/cache pose,
penalize torque/work,
reward staying near the target object position.
```

Finger/contact terms are logged for diagnosis, but they are not directly added as strong reward shaping. This avoids forcing unnatural contact behavior and keeps the learning target focused on stable rotation.

## Important Stability Fix

The hand pose penalty must compare against the reset/cache hand pose:

```text
current hand dof position - reset_hand_dof_pos
```

It should not compare against the Allegro default open pose. Comparing against the default open pose pushes the policy away from the stable cache grasp and often leads to thumb loss or two-finger-only rotation.

## Gravity Curriculum

Training starts with light gravity and ramps to full gravity. This lets the policy first learn the rolling motion from a stable grasp, then gradually learn to preserve the object under realistic load.

The final successful result reaches:

```text
gravity_z = -10.0000
drop_rate = 0.0000
mean episode length = 599.00
```

## Asset Handling

The task uses repo-relative assets:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/allegro_rotate/assets/allegro_hand_inst.usd
source/isaaclab_tasks/isaaclab_tasks/direct/allegro_rotate/assets/cylinder_nv_logo.usd
source/isaaclab_tasks/isaaclab_tasks/direct/allegro_rotate/assets/nvidia-logo-vert-rgb-blk-no-reg-for-screen.png
```

The cylinder USD owns its material. The task should not spawn a replacement `visual_material`, because that hides the USD-authored texture.

## Success Criteria

A good result should show:

```text
full-gravity stability,
zero or near-zero drop rate,
full episode length,
nonzero force on thumb/index/middle/ring,
object position staying close to reset/cache position,
visible rolling rotation without large off-axis wobble.
```

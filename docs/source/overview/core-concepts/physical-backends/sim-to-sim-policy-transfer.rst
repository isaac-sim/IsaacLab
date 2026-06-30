Transferring Policies Between Physics Backends
==============================================

Articulation ordering preserves name-to-vector semantics when a policy moves
between physics backends, but it does not make the backends' solver dynamics
identical. This guide uses ANYmal-D to replay one RSL-RL checkpoint on
Newton/MJWarp and PhysX without changing which physical joint or body each
vector element represents.


Quick Transfer
--------------

The examples use ``Isaac-Velocity-Flat-AnymalD`` for training and its ``-Play``
variant for replay. For general RSL-RL checkpoint and command options, see
:doc:`Reinforcement Learning Workflows <../../reinforcement-learning/rl_existing_scripts>`.

Newton/MJWarp to PhysX
^^^^^^^^^^^^^^^^^^^^^^

Train the source policy with Newton and the MJWarp solver:

.. code-block:: bash

    ./isaaclab.sh train --rl_library rsl_rl \
        --task Isaac-Velocity-Flat-AnymalD \
        --num_envs 4096 \
        physics=newton_mjwarp

Set ``CHECKPOINT`` to the absolute path of the intended checkpoint from that
Newton/MJWarp run, then replay it with PhysX:

.. code-block:: bash

    CHECKPOINT="/absolute/path/to/newton-mjwarp/model.pt"
    ./isaaclab.sh play --rl_library rsl_rl \
        --task Isaac-Velocity-Flat-AnymalD-Play \
        --checkpoint "${CHECKPOINT}" \
        physics=physx \
        env.scene.robot.joint_ordering=mjwarp \
        env.scene.robot.body_ordering=mjwarp

The ``mjwarp`` ordering value names the **source checkpoint semantics**. It
does not select the target backend; ``physics=physx`` does that.

PhysX to Newton/MJWarp
^^^^^^^^^^^^^^^^^^^^^^

Train the source policy with PhysX:

.. code-block:: bash

    ./isaaclab.sh train --rl_library rsl_rl \
        --task Isaac-Velocity-Flat-AnymalD \
        --num_envs 4096 \
        physics=physx

Set ``CHECKPOINT`` to the absolute path of the intended checkpoint from that
PhysX run, then replay it with Newton/MJWarp:

.. code-block:: bash

    CHECKPOINT="/absolute/path/to/physx/model.pt"
    ./isaaclab.sh play --rl_library rsl_rl \
        --task Isaac-Velocity-Flat-AnymalD-Play \
        --checkpoint "${CHECKPOINT}" \
        physics=newton_mjwarp \
        env.scene.robot.joint_ordering=physx \
        env.scene.robot.body_ordering=physx

Here, ``physx`` likewise names the source checkpoint semantics, while
``physics=newton_mjwarp`` selects the target backend.


Why Articulation Orders Differ
------------------------------

The same named articulation can have different native tensor-axis orders in
different physics backends.


Public and Backend Order
------------------------

Isaac Lab distinguishes the order exposed by the high-level articulation API
from the native order used by the active backend.


Conversion Cost
---------------

Selecting a nondefault public order can require conversion work around
backend reads and writes.


Direct Backend-View Access
--------------------------

Code that accesses ``root_view`` directly is responsible for respecting its
backend-native ordering.


What Ordering Does Not Solve
----------------------------

Stable vector semantics do not remove the physical differences between
solvers.


Verification and Troubleshooting
--------------------------------

When a transferred policy behaves unexpectedly, first separate ordering
mistakes from genuine solver-dynamics differences.

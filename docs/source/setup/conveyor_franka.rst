Conveyor Franka (Contrib)
=========================

Choose between the original four-cube racetrack task and the warehouse sorting task.
Both use the same pretrained Franka policy and shared manipulation code. The sorter extends
the base environment with textured cartons, elevated returns, gravity infeeds, and color dispatch.

.. image:: ../_static/conveyor_franka.jpg
   :alt: Franka sorting colored cartons between two conveyors in a warehouse
   :width: 100%

.. list-table:: Two tasks, one policy
   :header-rows: 1
   :widths: 22 24 54

   * - Task
     - Inventory
     - Behavior
   * - Racetrack transfer
     - Four numbered cubes
     - Original two closed racetracks; continuous alternating transfers.
   * - Warehouse sorting
     - 24 colored parcels
     - Extended circulating conveyors; blue/green on one loop, orange/purple on the other.

Both run on Newton GPU: select ``IsaacContrib-Conveyor-Franka-Newton-v0`` for the original
racetracks or ``IsaacContrib-Conveyor-Franka-Newton-Play-v0`` for sorting. The original task
also has a native PhysX CPU backend, described below. Sorting reuses the base configuration,
agent configuration, and manipulation terms; only the warehouse adds parcel-slot reassignment
and color-based dispatch.

Run the pretrained policy
-------------------------

Use the standard Isaac Lab installation with the Isaac Sim extra for Kit/RTX visuals:

.. code-block:: bash

   uv run --extra isaacsim isaaclab play --rl_library rsl_rl \
     --task IsaacContrib-Conveyor-Franka-Newton-Play-v0 \
     --checkpoint https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/IsaacContrib-Conveyor-Franka-Newton-v0_newtonmjwarp_none_rsl_rl.pt --num_envs 1 --device cuda:0 --viz kit --real-time \
     --kit_args=--/UJITSO/geometry=false

The first launch downloads the referenced Omniverse assets. The warehouse uses USD-authored
materials and lighting; Kit/RTX is the intended viewer. The launch override disables experimental
geometry streaming, including saved Kit preferences, which can hide meshes updated through Fabric.
To record this view, append ``--video --video_length 1440 env.sim.physics.use_cuda_graph=False``.
Disable CUDA graphs for this Kit recording path; compact training retains its graph-enabled default.

For compact, lightweight playback:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
     --task IsaacContrib-Conveyor-Franka-Newton-v0 \
     --checkpoint pretrained --num_envs 8 --device cuda:0 --viz newton_gl --real-time

The published policy is iteration 7998 of the
`conveyor training run <https://wandb.ai/nvidia-isaac/isaaclab-conveyor-franka/runs/conveyor-franka-resume-22gpu-model3999-exactsrc-pxr1-l40-20260810-3>`__.
The warehouse command uses that same checkpoint URL explicitly. Both retain 123 observations,
eight actions, 120 Hz physics, and a 60 Hz policy rate. Checkpoints remain outside the repository.

Train or compare backends
-------------------------

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Conveyor-Franka-Newton-v0 --num_envs 256 --device cuda:0

   uv run --extra isaacsim isaaclab play --rl_library rsl_rl \
     --task IsaacContrib-Conveyor-Franka-PhysX-CPU-v0 \
     --checkpoint /path/to/model.pt --num_envs 1 --device cpu --viz kit --real-time \
     agent.device=cpu

Native PhysX surface velocity requires CPU simulation for this task. GPU dynamics can drop
belt contacts in the supported Isaac Sim runtime. The PhysX variant rejects CUDA devices;
its separately named pretrained artifact is not published. Policy shape compatibility does
not imply identical behavior between backends.

Warehouse sorting
-----------------

The original manipulation straights and adjoining 90-degree bends remain fixed. Twenty-four
40 mm cartons circulate through the extended layout. Each reset shuffles a balanced batch of
six blue, six green, six orange, and six purple cartons. Blue/green belong on the positive-Y
loop; orange/purple belong on the negative-Y loop. Four policy slots are reassigned to arriving
parcels while an active grasp retains its identity. Destination classes are supervisory metadata;
the state-based checkpoint does not recognize colors from images.

This is a presentation and policy-reuse demonstration, not a reliably solved 24-parcel benchmark.
The unchanged checkpoint can miss grasps and reset before finishing a batch. The original
four-cube training and CPU reference configurations retain their compact layout.

.. raw:: html

   <video controls muted loop playsinline preload="metadata" style="width:100%;max-width:960px"
          poster="../../_static/conveyor_franka.jpg">
     <source src="https://github.com/maxkra15/IsaacLab/releases/download/conveyor-franka-preview/conveyor_franka.mp4" type="video/mp4">
     Your browser does not support embedded video.
   </video>

`Download the warehouse preview <https://github.com/maxkra15/IsaacLab/releases/download/conveyor-franka-preview/conveyor_franka.mp4>`__.

The task's
:download:`README <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/conveyor_franka/README.md>`
describes the USD assets, collision ownership, slot adapter, and sorting metrics.

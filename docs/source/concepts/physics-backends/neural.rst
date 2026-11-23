.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _neural-backend:

Neural (World Models)
=====================

The ``isaaclab_neural`` extension lets a learned **world model** replace the physics engine. A task keeps its
ordinary articulation and cameras; the world model supplies the joint state and the camera frames through the same
interfaces.

.. code-block:: text

    Policy / trainer  ->  Task  ->  Isaac Lab  ->  NeuralManager  ->  WorldModel

The world model only generates images. The manager keeps each environment's joint state itself (the reset record's
state at reset, then the last joint target), so joint observations and actions behave as on a physics backend.


Components
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Role
   * - ``NeuralCfg``
     - ``PhysicsCfg`` selected as the task's ``physics=`` preset. Holds the model configuration and the reset records.
   * - ``NeuralManager``
     - Physics backend (``backend_name = "neural"``). Buffers one joint target per step, calls the model once per
       chunk, and keeps the predicted frames of each view.
   * - ``WorldModel``
     - Abstract model: ``spec()``, ``reset()``, ``write()``, ``step()``, ``close()``.
   * - ``NeuralRendererCfg``
     - Selected as a camera's ``renderer_cfg``, with the model view the camera shows. Copies that view's frames into
       the camera, including its frame history.
   * - ``Articulation``
     - Virtual articulation. Shares the manager's joint state and targets, so the task's joint observation and action
       terms work unchanged. Its reset starts an episode from a reset record. Everything that needs a body raises.
   * - ``at_chunk_boundary``
     - Termination term that lets a condition fire only where environments can reset.
   * - ``Cosmos3Cfg``, ``Cosmos3Model``
     - Client of the Cosmos3 world-model service (``isaaclab_neural.models.cosmos3``).


Using a task's configuration
----------------------------

The task declares an ``ArticulationCfg`` and a ``CameraCfg`` as it would for a physics backend, and selects the world
model with ``NeuralCfg(model_cfg=..., reset_manifest=...)`` and ``NeuralRendererCfg()``. The camera needs
``frame_history_length`` equal to the model's chunk length and the model's image size.

* **Views**: a model predicts one or more named camera views (``WorldModelSpec.views``). Each world-model camera shows
  one of them, set with ``NeuralRendererCfg(view=...)``; for a model with one view, ``view`` can stay ``None``.
  Several cameras of a multi-view model each select their own view.
* **Outputs**: a model predicts one or more camera outputs per view (``WorldModelSpec.outputs``, default
  ``("rgb",)``), named as camera data types. A camera requests them in ``data_types``, and
  ``NeuralRendererCfg.outputs`` declares their layout, RGB by default; for depth, for example, add
  ``RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32)``.
* **Joints and limits** come from the spawned USD asset (a URDF spawner converts to USD): the movable joints in
  traversal order, or in the order of ``ArticulationCfg.joint_ordering``, which must match the reset records. Joint
  limits are read from the USD, and ``soft_joint_pos_limit_factor`` applies as usual. Movable joints are revolute or
  prismatic; a joint with several axes, such as a spherical or D6 joint, raises.
* **Observations and actions**: ``joint_pos`` terms read the joint state and ``JointPositionAction`` writes the target.
  Velocity and effort targets, poses, forces and physical properties raise ``NotImplementedError``.
* **Resets**: ``Articulation.reset``, which the scene calls before the reset events, samples a reset record
  (``image``, ``state``, ``prompt``, ``split`` in the JSON manifest) and starts the episode from it: the record's start
  image becomes the camera's frames (``image`` is a path, ``{view: path}``, or ``{view: {output: path}}`` with ``.npy``
  arrays for outputs other than RGB), and its joint state the joint state and the default joint position. The model
  restarts the environment from the record's image and prompt at the end of the next chunk, before it predicts that
  chunk.

  The standard ``reset_joints_by_offset`` and ``reset_joints_by_scale`` events therefore start from the record; a
  non-zero range moves the joints away from the start image the model sees, so randomize through the records instead.
  ``reset_scene_to_default``, the default reset event, also resets the root pose, which the world model does not have,
  so a task replaces the default event configuration. Training samples records at random; other splits visit them in
  order. ``envs_per_reset_record`` consecutive environments share a record (default 1); a group-relative algorithm
  such as GRPO sets it to its group size so that a group compares rollouts of one scene.
* **Device**: joint state, targets, frames and the start images of the split's records stay on the simulation device.
  Resets and the steps inside a chunk read nothing back to the host. The step that ends a chunk, which waits for the
  model anyway, reads which environments reset, since the records' prompts are host strings.


Timing
------

A model predicts ``chunk_length`` frames from ``chunk_length`` joint targets. Use ``sim.dt = 1 / fps`` and a
``decimation`` that divides the chunk length; with ``decimation = 1``, one environment step is one frame. The model
runs on the step that completes a chunk, and the frames of a chunk arrive together, so a reward on them is paid in
that step.

Isaac Lab resets an environment in any step in which a termination fires, but the model can only restart between
chunks, so a reset in the middle of a chunk raises when the chunk ends. Run every termination, the time-out too,
through ``at_chunk_boundary``, with the condition and its parameters as the term's parameters:

.. code-block:: python

    done = DoneTerm(func=at_chunk_boundary, params={"term": my_condition, "term_params": {...}})
    time_out = DoneTerm(func=at_chunk_boundary, params={"term": mdp.time_out}, time_out=True)

* The condition is judged on the step that completes a chunk. A condition that stays true, such as a success, a
  failure or the time limit, is caught at the next boundary, at most one chunk late; one that holds only briefly
  inside a chunk is missed.
* Episodes can therefore have any length, including the random first episode lengths some trainers use
  (rsl_rl's ``init_at_random_ep_len``).
* Each environment resets at its own next boundary; it does not wait for the others.


Add a world model
-----------------

Implement the ``WorldModel`` interface, subclass ``ModelCfg`` and set its ``class_type``. The manager builds the model
as ``class_type(cfg, num_envs, device)`` with the simulation device, and the model keeps what it needs of those:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Responsibility
   * - ``spec()``
     - Chunk length, frame rate, image size, the largest number of environments, the inputs it needs, and the names
       of the views and outputs.
   * - ``reset(env_ids, images, prompts, seeds)``
     - Restart the selected environments (host lists) from their start frames, ``{view: {output: (K, H, W, C)}}``.
       Called right before ``step()``, for the environments that reset since the last chunk.
   * - ``write(name, value)``
     - Set one of the inputs the spec lists, named by ``WorldModelAttribute``: the chunk's joint targets
       ``(N, T, J)`` (``JOINT_TARGETS``) or the joint state before the chunk ``(N, J)`` (``START_JOINT_STATE``).
       Called before every ``step()``.
   * - ``step()``
     - Predict the next chunk from the inputs written, as frames ``{view: {output: (N, T, H, W, C)}}``.
   * - ``close()``
     - Release resources.

Frames and inputs are torch tensors on the device. Each written input is the manager's buffer, reused for the next
chunk, so a model that keeps it copies it. A new input needs the backend to provide it as a ``WorldModelAttribute``;
the manager rejects a model whose spec needs one it does not provide. Construction is cheap; load the weights on the
first call.



Cosmos3
-------

Cosmos3 runs in its own process, in the Cosmos Framework's environment, so Isaac Lab does not install the Framework.
``Cosmos3Model`` is the client: it reads the chunk length, frame rate and image size from the service, sends start
images and controls, and receives frames. The service is the world-model counterpart of the Cosmos Transfer service
(``source/isaaclab_contrib/docs/cosmos_transfer.md``) and uses the same transport.

In a server terminal, create the Framework environment at commit ``3a84cd3b67f60520f523e184fa7b82011bc75cb4`` of
`cosmos-framework <https://github.com/NVIDIA/cosmos-framework>`_, install the Cosmos service package into it, and
start the service with a ``unified_v1`` checkpoint:

.. code-block:: bash

    cd "$COSMOS_FRAMEWORK_ROOT"
    uv sync --locked --all-extras --group=cu130-train
    uv run --no-sync uv pip install --editable "$ISAAC_LAB_ROOT/source/isaaclab_cosmos_server"
    export LD_LIBRARY_PATH=
    uv run --no-sync isaaclab-cosmos-world-model-server --checkpoint "$COSMOS3_CHECKPOINT"

Wait for ``Cosmos ready at ...``, then run the task in the Isaac Lab environment. A task selects the model with
``NeuralCfg(model_cfg=Cosmos3Cfg(joint_mapping=...))``. ``Cosmos3Model.step`` converts joint targets to 59
normalized controls with the task's ``JointMappingCfg``, so the robot description and the normalization stay with the
task.

The service's ``--mode window`` (default) generates any number of environments as one batch, each with its own start
frame, prompt and seed; ``--mode session`` serves one. A larger batch is faster per environment, and GPU memory limits
it; ``scripts/benchmarks/benchmark_cosmos3_world_model.py``, run in the Framework environment, measures both for each
batch size. With the service on the same machine and a GPU this process can access directly, chunks stay on the GPU
(CUDA IPC, Linux); otherwise they go through the socket, which carries about 16 environments of 16 frames at
480 x 640 per message. ``Cosmos3Cfg.transport`` forces either one.

Several GPUs
^^^^^^^^^^^^

Several services can share the environments, for example one service per GPU, as for Cosmos Transfer. Start each
service in its own terminal with its own GPU and endpoint; each loads its own copy of the model:

.. code-block:: bash

    cd "$COSMOS_FRAMEWORK_ROOT"
    export LD_LIBRARY_PATH=
    uv run --no-sync isaaclab-cosmos-world-model-server --checkpoint "$COSMOS3_CHECKPOINT" --device cuda:1       --endpoint unix:///tmp/isaaclab-cosmos3-gpu1.sock

Start the second with ``--device cuda:2 --endpoint unix:///tmp/isaaclab-cosmos3-gpu2.sock``, and set
``Cosmos3Cfg.endpoint`` to the list of both:

.. code-block:: python

    Cosmos3Cfg(endpoint=["unix:///tmp/isaaclab-cosmos3-gpu1.sock", "unix:///tmp/isaaclab-cosmos3-gpu2.sock"], ...)

The environments are split into contiguous shares of near-equal size, one per service in list order: with 4
environments, 0-1 on the first service and 2-3 on the second. Every chunk sends each service its share before
waiting for any reply, so the services generate at the same time, and a chunk lasts as long as the slowest service.
There must be at least one environment per service. If one service fails, the model closes every session.


Tests
-----

The tests use a fake model, with no weights, GPU or simulation app:

.. code-block:: bash

    uv run python -m pytest source/isaaclab_neural/test -q

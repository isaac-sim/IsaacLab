IO Descriptors 101
==================

In this tutorial, we will learn about IO descriptors, what they are, how to export them, and how to add them to
your environments. We will use the Anymal-D robot as an example to demonstrate how to export IO descriptors from
an environment, and use our own terms to demonstrate how to attach IO descriptors to custom action and observation terms.


What are IO Descriptors?
------------------------

Before we dive into IO descriptors, let's first understand what they are and how they can be useful.

IO descriptors are a way to describe the inputs and outputs of a policy trained using the ManagerBasedRLEnv in Isaac
Lab. In other words, they describe the action and observation terms of a policy. This description is used to generate
a YAML file that can be loaded in an external tool to run the policies without having to manually input the
configuration of the action and observation terms.

In addition to this the IO Descriptors provide the following information:
- The parameters of all the joints in the articulation.
- Some simulation parameters including the simulation time step, and the policy time step.
- For some action and observation terms, it provides the joint names or body names in the same order as they appear in the action/observation terms.
- For both the observation and action terms, it provides the terms in the exact same order as they appear in the managers. Making it easy to
  reconstruct them from the YAML file.

Here is an example of what the action part of the YAML generated from the IO descriptors looks like for the Anymal-D robot:
.. literalinclude:: ../../_static/policy_deployment/01_io_descriptors/isaac_velocity_flat_anymal_d_v0_IO_descriptors.yaml
   :language: yaml
   :lines: 1-39

Here is an example of what a portion of the observation part of the YAML generated from the IO descriptors looks like for the Anymal-D robot:
.. literalinclude:: ../../_static/policy_deployment/01_io_descriptors/isaac_velocity_flat_anymal_d_v0_IO_descriptors.yaml
   :language: yaml
   :lines: 158-199

.. literalinclude:: ../../_static/policy_deployment/01_io_descriptors/isaac_velocity_flat_anymal_d_v0_IO_descriptors.yaml
   :language: yaml
   :lines: 236-279

Something to note here is that both the action and observation terms are returned as list of dictionaries, and not a dictionary of dictionaries.
This is done to ensure the order of the terms is preserved. Hence, to retrieve the action or observation term, the users need to look for the
``name`` key in the dictionaries.

For example, in the following snippet, we are looking at the ``projected_gravity`` observation term. The ``name`` key is used to identify the term.
The ``full_path`` key is used to provide an explicit path to the function in Isaac Lab's source code that is used to compute this term. Some flags
like ``mdp_type`` and ``observation_type`` are also provided, these don't have any functional impact. They are here to inform the user that this is the
category this term belongs to.

.. literalinclude:: ../../_static/policy_deployment/01_io_descriptors/isaac_velocity_flat_anymal_d_v0_IO_descriptors.yaml
   :language: yaml
   :lines: 200-219
   :emphasize-lines: 211,209


Exporting IO Descriptors from an Environment
--------------------------------------------

In this section, we will cover how to export IO descriptors from an environment.
Keep in mind that this feature is only available to the manager based RL environments.

If a policy has already been trained using a given configuration, then the IO descriptors can be exported using:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/export_io_descriptors.py --task <task_name> --output_dir <output_dir>

For example, if we want to export the IO descriptors for the Anymal-D robot, we can run:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/export_io_descriptors.py --task Isaac-Velocity-Flat-Anymal-D-v0 --output_dir ./io_descriptors

When training a policy, it is also possible to request the IO descriptors to be exported at the beginning of the training.
This can be done by setting the ``export_io_descriptors`` flag in the command line.

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --export_io_descriptors
   ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --export_io_descriptors
   ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --export_io_descriptors
   ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --export_io_descriptors


Attaching IO Descriptors to Custom Observation Terms
---------------------------------------------------

In this section, we will cover how to attach IO descriptors to custom observation terms.

Let's take a look at how we can attach an IO descriptor to a simple observation term:

.. code-block:: python
   @generic_io_descriptor(
      units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
   )
   def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
      """Root linear velocity in the asset's root frame."""
      # extract the used quantities (to enable type-hinting)
      asset: RigidObject = env.scene[asset_cfg.name]
      return asset.data.root_lin_vel_b

Here, we are defining a custom observation term called ``base_lin_vel`` that computes the root linear velocity of the robot.
We are also attaching an IO descriptor to this term. The IO descriptor is defined using the ``@generic_io_descriptor`` decorator.

The ``@generic_io_descriptor`` decorator is a special decorator that is used to attach an IO descriptor to a custom observation term.
It takes arbitrary arguments that are used to describe the observation term, in this case we provide extra information that could be
useful for the end user:

- ``units``: The units of the observation term.
- ``axes``: The axes of the observation term.
- ``observation_type``: The type of the observation term.

You'll also notice that there is an ``on_inspect`` argument that is provided. This is a list of functions that are used to inspect the observation term.
In this case, we are using the ``record_shape`` and ``record_dtype`` functions to record the shape and dtype of the output of the observation term.

These functions are defined like so:
.. code-block:: python
   def record_shape(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
      """Record the shape of the output tensor.

      Args:
         output: The output tensor.
         descriptor: The descriptor to record the shape to.
         **kwargs: Additional keyword arguments.
      """
      descriptor.shape = (output.shape[-1],)


   def record_dtype(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
      """Record the dtype of the output tensor.

      Args:
         output: The output tensor.
         descriptor: The descriptor to record the dtype to.
         **kwargs: Additional keyword arguments.
      """
      descriptor.dtype = str(output.dtype)

They always take the output tensor of the observation term as the first argument, and the descriptor as the second argument.
In the ``kwargs`` all the inputs of the observation term are provided.

Let us now take a look at a more complex example: getting the relative joint positions of the robot.
.. code-block:: python
   @generic_io_descriptor(
       observation_type="JointState",
       on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
       units="rad",
   )
   def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
       """The joint positions of the asset w.r.t. the default joint positions.

       Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
       """
       # extract the used quantities (to enable type-hinting)
       asset: Articulation = env.scene[asset_cfg.name]
       return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

Similarly to the previous example, we are adding an IO descriptor to a custom observation term with a set of functions that probe the observation term.

To get the name of the joints we can write the following function:

.. code-block:: python
   def record_joint_names(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
       """Record the joint names of the output tensor.

       Expects the `asset_cfg` keyword argument to be set.

       Args:
           output: The output tensor.
           descriptor: The descriptor to record the joint names to.
           **kwargs: Additional keyword arguments.
       """
       asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
       joint_ids = kwargs["asset_cfg"].joint_ids
       if joint_ids == slice(None, None, None):
           joint_ids = list(range(len(asset.joint_names)))
       descriptor.joint_names = [asset.joint_names[i] for i in joint_ids]

Note that we can access all the inputs of the observation term in the ``kwargs`` dictionary. Hence we can access the ``asset_cfg``, which contains the
configuration of the articulation that the observation term is computed on.

To get the offsets, we can write the following function:
.. code-block:: python
   def record_joint_pos_offsets(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs):
    """Record the joint position offsets of the output tensor.

    Expects the `asset_cfg` keyword argument to be set.

    Args:
        output: The output tensor.
        descriptor: The descriptor to record the joint position offsets to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    ids = kwargs["asset_cfg"].joint_ids
    # Get the offsets of the joints for the first robot in the scene.
    # This assumes that all robots have the same joint offsets.
    descriptor.joint_pos_offsets = asset.data.default_joint_pos[:, ids][0]

With this in mind, you should now be able to attach an IO descriptor to your own custom observation terms! However, before
we close this tutorial, let's take a look at how we can attach an IO descriptor to a custom action term.


Attaching IO Descriptors to Custom Action Terms
----------------------------------------------

In this section, we will cover how to attach IO descriptors to custom action terms.

IO Descriptors 101
==================

.. currentmodule:: isaaclab

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
- For both the observation and action terms, it provides the terms in the exact same order as they appear in the managers. Making it easy to reconstruct them from the YAML file.

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
   :emphasize-lines: 9, 11


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
----------------------------------------------------

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
In the ``kwargs`` all the inputs of the observation term are provided. In addition to the ``on_inspect`` functions, the decorator
will also call call some functions in the background to collect the ``name``, the ``description``, and the ``full_path`` of the
observation term. Note that adding this decorator does not change the signature of the observation term, so it can be used safely
with the observation manager!

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
-----------------------------------------------

In this section, we will cover how to attach IO descriptors to custom action terms. Action terms are classes that
inherit from the :class:`managers.ActionTerm` class. To add an IO descriptor to an action term, we need to expand
upon its :meth:`ActionTerm.IO_descriptor` property.

By default, the :meth:`ActionTerm.IO_descriptor` property returns the base descriptor and fills the following fields:
- ``name``: The name of the action term.
- ``full_path``: The full path of the action term.
- ``description``: The description of the action term.
- ``export``: Whether to export the action term.

.. code-block:: python

   @property
   def IO_descriptor(self) -> GenericActionIODescriptor:
       """The IO descriptor for the action term."""
       self._IO_descriptor.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()
       self._IO_descriptor.full_path = f"{self.__class__.__module__}.{self.__class__.__name__}"
       self._IO_descriptor.description = " ".join(self.__class__.__doc__.split())
       self._IO_descriptor.export = self.export_IO_descriptor
       return self._IO_descriptor

To add more information to the descriptor, we need to override the :meth:`ActionTerm.IO_descriptor` property.
Let's take a look at an example on how to add the joint names, scale, offset, and clip to the descriptor.

.. code-block:: python

   @property
   def IO_descriptor(self) -> GenericActionIODescriptor:
       """The IO descriptor of the action term.

       This descriptor is used to describe the action term of the joint action.
       It adds the following information to the base descriptor:
       - joint_names: The names of the joints.
       - scale: The scale of the action term.
       - offset: The offset of the action term.
       - clip: The clip of the action term.

       Returns:
           The IO descriptor of the action term.
       """
       super().IO_descriptor
       self._IO_descriptor.shape = (self.action_dim,)
       self._IO_descriptor.dtype = str(self.raw_actions.dtype)
       self._IO_descriptor.action_type = "JointAction"
       self._IO_descriptor.joint_names = self._joint_names
       self._IO_descriptor.scale = self._scale
       # This seems to be always [4xNum_joints] IDK why. Need to check.
       if isinstance(self._offset, torch.Tensor):
           self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
       else:
           self._IO_descriptor.offset = self._offset
       # FIXME: This is not correct. Add list support.
       if self.cfg.clip is not None:
           if isinstance(self._clip, torch.Tensor):
               self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist()
           else:
               self._IO_descriptor.clip = self._clip
       else:
           self._IO_descriptor.clip = None
       return self._IO_descriptor

This is it! You should now be able to attach an IO descriptor to your own custom action terms which concludes this tutorial.

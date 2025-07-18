IO Descriptors 101
==================

In this tutorial, we will learn about IO descriptors, what they are, how to export them, and how to add them to your environment.
We will use the Anymal-D robot as an example to demonstrate how to export IO descriptors an environment, and the Spot robot to
explain how to attach IO descriptors to custom action and observation terms.


What are IO Descriptors?
------------------------

Before we dive into IO descriptors, let's first understand what they are and why they are useful.

IO descriptors are a way to describe the input and output of a policy trained using the ManagerBasedRLEnv in Isaac Lab. In other words,
they describe the action and observation terms of a policy. This description is used to generate a YAML file that can be loaded in an
external tool to run the policies without having to manually input the configuration of the action and observation terms.

In addition to this the IO Descriptors provide the following information:
- The parameters of all the joints in the articulation.
- Some simulation parameters including the simulation time step, and the policy time step.
- For some action and observation terms, it provides the joint names or body names in the same order as they appear in the action terms.
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
This is done to ensure the order of the terms is preserved. Hence, to retrive the action or observation term, the users need to look for the
`name` key in the dictionaries.

For example, in the following snippet, we are looking at the `projected_gravity` observation term. The `name` key is used to identify the term.
The `full_path` key is used to provide an explicit path to the function in Isaac Lab's source code. Some flags like `mdp_type` and `observation_type`
are also provided, these don't have any functional impact. They are here to inform the user that this is the category this term belongs to.

.. literalinclude:: ../../_static/policy_deployment/01_io_descriptors/isaac_velocity_flat_anymal_d_v0_IO_descriptors.yaml
   :language: yaml
   :lines: 200-219
   :emphasize-lines: 211,209




Exporting IO Descriptors from an Environment
--------------------------------------------




IO descriptors are a way to describe the input and output of a policy.
They are used to describe the action and observation spaces of a policy.icy.
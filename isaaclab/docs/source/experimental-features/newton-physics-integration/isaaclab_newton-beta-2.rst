Isaac Lab - Newton Beta 2
=========================

Isaac Lab - Newton Beta 2 (feature/newton branch) provides Newton physics engine integration for Isaac Lab. We refactored our code so that we can not only support PhysX and Newton, but
any other physics engine, enabling users to bring their own physics engine to Isaac Lab if they desire. To enable this, we introduce base implementations of
our ``simulation interfaces``, :class:`~isaaclab.assets.articulation.Articulation` or :class:`~isaaclab.sensors.ContactSensor` for instance. These provide a
set of abstract methods that all physics engines must implement. In turn this allows all of the default Isaac Lab environments to work with any physics engine.
This also allows us to ensure that Isaac Lab - Newton Beta 2 is backwards compatible with Isaac Lab 2.X. For engine specific calls, users could get the underlying view of
the physics engine and call the engine specific APIs directly.

However, as we are refactoring the code, we are also looking at ways to limit the overhead of Isaac Lab's. In an effort to minimize the overhead, we are moving
all our low level code away from torch, and instead will rely heavily on warp. This will allow us to write low level code that is more efficient, and also
to take advantage of the cuda-graphing. However, this means that the ``data classes`` such as :class:`~isaaclab.assets.articulation.ArticulationData` or
:class:`~isaaclab.sensors.ContactSensorData` will only return warp arrays. Users will hence have to call ``wp.to_torch`` to convert them to torch tensors if they desire.
Our setters/writers will support both warp arrays and torch tensors, and will use the most optimal strategy to update the warp arrays under the hood. This minimizes the
amount of changes required for users to migrate to Isaac Lab - Newton Beta 2.

Another new feature of the writers and setters is the ability to provide them with masks and complete data (as opposed to indices and partial data in Isaac Lab 2.X).
Note that this feature will be available along with the ability to provide indices and partial data, and that the default behavior will still be to provide indices and partial data.
However, if using warp, users will have to provide masks and complete data. In general we encourage users to move to adopt this new feature as, if done well, it will
reduce on the fly memory allocations, and should result in better performance.

On the optimization front, we decided to change quaternion conventions. Originally, Isaac Lab and Isaac Sim both adopted the ``wxyz`` convention. However, we were doing several
conversions to and from ``xyzw`` in our setters/writers as PhysX uses the ``xyzw`` convention. Since both Newton and Warp, also use the ``xyzw`` convention, we decided to change
our default convention to ``xyzw``. This means that all our APIs will now return quaternions in the ``xyzw`` convention. This is likely a breaking change for all the custom
mdps that are not using our :mod:`~isaaclab.utils.math` module. While this change is substantial, it should make things more consistent for when users are using the simulation
views directly, and will remove needless conversions.

Finally, alongside the new isaaclab_newton extension, we are also introducing new isaaclab_experimental and isaaclab_task_experimental extensions. These extensions will allow
us to quickly bring new features to Isaac Lab main while giving them the time they need to mature before being fully integrated into the core Isaac Lab extensions. In this release,
we are introducing cuda-graphing support for direct rl tasks. This drastically reduces Isaac Lab's overhead making training faster. Try them out and let us know what you think!

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-Direct-Warp-v0 --num_envs 4096 --headless

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Direct-Warp-v0 --num_envs 4096 --headless

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-Direct-Warp-v0 --num_envs 4096 --headless


What's Next?
============

Isaac Lab 3.0 is the upcoming release of Isaac Lab, which will be compatible with Isaac Sim 6.0, and at the same time will support the new Newton physics engine.
This will allow users to train policies on the Newton physics engine, or PhysX. To accommodate this major code refactoring are required. In this section, we
will go over some of the changes, how that will affect Isaac Lab 2.X users, and how to migrate to Isaac Lab 3.0. The current branch of ``feature/newton`` gives
a glance of what is to come. While the changes to the internal code structure are significant, the changes to the user API are minimal.

Newton Physics Integration
===========================

`Newton <https://newton-physics.github.io/newton/latest/guide/overview.html>`_ is a GPU-accelerated, extensible, and differentiable physics simulation engine designed for robotics, research,
and advanced simulation workflows. Built on top of `NVIDIA Warp <https://nvidia.github.io/warp/>`_ and integrating MuJoCo Warp, Newton provides high-performance
simulation, modern Python APIs, and a flexible architecture for both users and developers.

Newton is an Open Source community-driven project with contributions from NVIDIA, Google Deep Mind, and Disney Research,
managed through the Linux Foundation.

This integration is available on the `develop branch <https://github.com/isaac-sim/IsaacLab/tree/develop>`_ of Isaac Lab as part of Isaac Lab 3.0 Beta, and is
under active development. Many features are not yet supported, and only a limited set of classic RL and flat terrain locomotion
reinforcement learning examples are included at the moment.

Both this Isaac Lab integration and Newton itself are under heavy development. We intend to support additional
features for other reinforcement learning and imitation learning workflows in the future, but the above tasks should be
a good lens through which to understand how Newton integration works in Isaac Lab.

We have validated Newton simulation against PhysX by transferring learned policies from Newton to PhysX and vice versa
Furthermore, we have also successfully deployed a Newton-trained locomotion policy to a G1 robot. Please see :ref:`here <sim2real>` for more information.

Newton can support `multiple solvers <https://newton-physics.github.io/newton/latest/api/newton_solvers.html>`_ for handling different types of physics simulation, but for the moment, the Isaac
Lab integration focuses primarily on the MuJoCo-Warp solver.

Future updates of Isaac Lab and Newton should include both ongoing improvements in performance as well as integration
with additional solvers.

During the development phase of both Newton and this Isaac Lab integration, you are likely to encounter breaking
changes as well as limited documentation. We do not expect to be able to provide official support or debugging assistance
until the framework has reached an official release. We appreciate your understanding and patience as we work to deliver a robust and polished framework!

For an overview of how the multi-backend architecture works, including how to add a new backend, see
:doc:`/source/overview/core-concepts/multi_backend_architecture`.


.. toctree::
  :maxdepth: 2
  :titlesonly:

  installation
  isaaclab_newton-beta-2
  training-environments
  visualization
  limitations-and-known-bugs
  solver-transitioning
  sim-to-sim
  sim-to-real

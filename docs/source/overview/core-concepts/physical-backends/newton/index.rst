Newton Backend
==============

`Newton <https://newton-physics.github.io/newton/latest/guide/overview.html>`_ is a
GPU-accelerated, extensible, and differentiable physics simulation engine designed
for robotics, research, and advanced simulation workflows. Built on top of
`NVIDIA Warp <https://nvidia.github.io/warp/>`_ and integrating MuJoCo Warp, Newton
provides high-performance simulation, modern Python APIs, and a flexible
architecture for both users and developers.

Newton is an Open Source community-driven project with contributions from NVIDIA,
Google Deep Mind, and Disney Research, managed through the Linux Foundation.

Newton support in Isaac Lab is in beta and under active development. Many features
are still maturing, and the Isaac Lab integration ships a focused, validated set of
classic RL and flat-terrain locomotion environments. We have validated Newton
simulation against PhysX by transferring learned policies in both directions and
have successfully deployed a Newton-trained locomotion policy to a G1 robot.

Newton can support `multiple solvers
<https://newton-physics.github.io/newton/latest/api/newton_solvers.html>`_ for
handling different types of physics simulation. The Isaac Lab integration ships
the following solver pages:

* :doc:`mjwarp-solver` — the primary, validated solver path.
* :doc:`kamino-solver` — beta support on selected classic tasks.
* :doc:`using-vbd-solver` — experimental VBD solver for cloth and soft bodies,
  available through :mod:`isaaclab_contrib.deformable` and the MJWarp + VBD or
  Featherstone + VBD coupled managers.

Each solver is exposed as a small subclass of
:class:`~isaaclab_newton.physics.NewtonManager`. See
:doc:`newton-manager-abstraction` for the developer-facing guide to adding a
new solver or a coupled solver.

During the beta phase, breaking changes and incomplete documentation are still
expected. Official support and debugging assistance will follow once the framework
reaches an official release.

For an overview of how the multi-backend architecture works, including how to add a
new backend, see :doc:`../../multi_backend_architecture`.


.. toctree::
  :maxdepth: 2
  :titlesonly:

  installation
  supported-features
  mjwarp-solver
  kamino-solver
  using-vbd-solver
  newton-manager-abstraction
  warp-environments
  warp-env-migration

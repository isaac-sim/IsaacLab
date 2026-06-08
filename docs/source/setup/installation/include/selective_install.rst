Modularized Installation
~~~~~~~~~~~~~~~~~~~~~~~~

``./isaaclab.sh -i`` (or ``isaaclab.bat -i``) **always** installs all core
source packages. Additional arguments can control which **optional** submodules and
**extra feature dependencies** are installed.

**Default behavior:** ``./isaaclab.sh --install`` or ``./isaaclab.sh -i``
are equivalent to ``./isaaclab.sh -i all``.
That installs optional submodules (``mimic``, ``teleop``) and automatic extra
features (``newton``, ``rl``, ``visualizer``) on top of the core set. It does
**not** install ``contrib`` or ``ov`` runtime wheels or the Isaac Sim pip
package—request those explicitly when needed.

**Special values:**

- ``all`` — same as the default behavior above.
- ``core`` — core packages only; no optional submodules, no extra feature
  dependencies.

**Optional submodules** (installed only when requested or with ``all``):

.. list-table::
   :header-rows: 1

   * - Token
     - What it installs
   * - ``mimic``
     - ``isaaclab_teleop`` and ``isaaclab_mimic`` (imitation-learning tools)
   * - ``teleop``
     - ``isaaclab_teleop`` only (Linux x86_64)

**Extra feature sets** (optional heavy dependencies on core packages):

.. list-table::
   :header-rows: 1

   * - Token
     - What it installs
   * - ``newton``
     - Newton physics dependencies on ``isaaclab_newton``, ``isaaclab_physx``,
       and ``isaaclab_visualizers`` (selectors are not supported)
   * - ``rl[<framework>]``
     - RL framework extras on ``isaaclab_rl``. Selectors: ``rsl-rl``, ``skrl``,
       ``sb3``, ``rl-games``. Omit the selector to install all frameworks.
   * - ``visualizer[<backend>]``
     - Visualizer backend extras on ``isaaclab_visualizers``. Selectors:
       ``rerun``, ``viser``, ``newton``, ``kit``. Omit the selector for all
       backends.
   * - ``contrib[<feature>]``
     - Contrib runtime extras. Selector: ``rlinf``. Bare ``contrib`` installs
       no additional dependencies (the source package is already in core).
   * - ``ov[<runtime>]``
     - OV runtime wheels. Selectors: ``ovrtx``, ``ovphysx``, ``all``. Bare
       ``ov`` installs no additional dependencies (the source packages are
       already in core).
   * - ``isaacsim``
     - Isaac Sim pip package (via the install script; use only when Isaac Sim
       is not already installed)

``contrib`` and ``ov`` are **not** part of the default ``all`` install. Request
them explicitly when you need rlinf, OVRTX, or OVPhysX runtimes.

Pass a comma-separated list to combine tokens (commas inside ``[...]`` are
preserved). Unknown tokens emit a warning and are skipped.

Examples:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         # Default: core + optional submodules + newton/rl/visualizer extras
         ./isaaclab.sh -i

         # Newton physics + RSL-RL (common kit-less setup)
         ./isaaclab.sh -i 'newton,rl[rsl-rl]'

         # Newton + OVRTX renderer + RSL-RL + Newton visualizer
         ./isaaclab.sh -i 'newton,ov[ovrtx],rl[rsl-rl],visualizer[newton]'

         # Contrib rlinf runtime dependencies
         ./isaaclab.sh -i 'contrib[rlinf]'

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         :: Default: core + optional submodules + newton/rl/visualizer extras
         isaaclab.bat -i

         :: Newton physics + RSL-RL
         isaaclab.bat -i "newton,rl[rsl-rl]"

         :: Newton + OVRTX + RSL-RL + Newton visualizer
         isaaclab.bat -i "newton,ov[ovrtx],rl[rsl-rl],visualizer[newton]"

         :: Contrib rlinf runtime dependencies
         isaaclab.bat -i "contrib[rlinf]"

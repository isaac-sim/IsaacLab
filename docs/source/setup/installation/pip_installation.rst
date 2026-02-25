.. _isaaclab-pip-installation:

Installation using Isaac Sim Pip Package
========================================

The following steps first installs Isaac Sim from pip, then Isaac Lab from source code.

.. attention::

   Installing Isaac Sim with pip requires GLIBC 2.35+ version compatibility.
   To check the GLIBC version on your system, use command ``ldd --version``.

   This may pose compatibility issues with some Linux distributions. For instance, Ubuntu 20.04 LTS
   has GLIBC 2.31 by default. If you encounter compatibility issues, we recommend following the
   :ref:`Isaac Sim Binaries Installation <isaaclab-binaries-installation>` approach.

.. note::

   If you plan to :ref:`Set up Visual Studio Code <setup-vs-code>` later, we recommend following the
   :ref:`Isaac Sim Binaries Installation <isaaclab-binaries-installation>` approach.

Installing Isaac Sim
--------------------

From Isaac Sim 4.0 onwards, it is possible to install Isaac Sim using pip.
This approach makes it easier to install Isaac Sim without requiring to download the Isaac Sim binaries.
If you encounter any issues, please report them to the
`Isaac Sim Forums <https://docs.isaacsim.omniverse.nvidia.com/latest/common/feedback.html>`_.

.. attention::

   On Windows, it may be necessary to `enable long path support <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later>`_
   to avoid installation errors due to OS limitations.

.. include:: include/pip_python_virtual_env.rst

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   In case you used UV to create your virtual environment, please replace ``pip`` with ``uv pip``
   in the following commands.

-  Install Isaac Sim pip packages:

   .. code-block:: none

      pip install "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com

-  Install a CUDA-enabled PyTorch build that matches your system architecture:

   .. tab-set::
      :sync-group: pip-platform

      .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
         :sync: linux-x86_64

         .. code-block:: bash

            pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

      .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
         :sync: windows-x86_64

         .. code-block:: bash

            pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

      .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
         :sync: linux-aarch64

         .. code-block:: bash

            pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130

         .. note::

            After installing Isaac Lab on aarch64, you may encounter warnings such as:

            .. code-block:: none

               ERROR: ld.so: object '...torch.libs/libgomp-XXXX.so.1.0.0' cannot be preloaded: ignored.

            This occurs when both the system and PyTorch ``libgomp`` (GNU OpenMP) libraries are preloaded.
            Isaac Sim expects the **system** OpenMP runtime, while PyTorch sometimes bundles its own.

            To fix this, unset any existing ``LD_PRELOAD`` and set it to use the system library only:

            .. code-block:: bash

               unset LD_PRELOAD
               export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

            This ensures the correct ``libgomp`` library is preloaded for both Isaac Sim and Isaac Lab,
            removing the preload warnings during runtime.

.. include:: include/pip_verify_isaacsim.rst

Installing Isaac Lab
--------------------

.. include:: include/src_clone_isaaclab.rst

.. include:: include/src_build_isaaclab.rst

.. include:: include/src_verify_isaaclab.rst

.. _template-generator:

Build your Own Project or Task
==============================

Traditionally, building new projects that utilize Isaac Lab's features required creating your own
extensions within the Isaac Lab repository. However, this approach can obscure project visibility and
complicate updates from one version of Isaac Lab to another. To circumvent these challenges,
we now provide a command-line tool (**template generator**) for creating Isaac Lab-based projects and tasks.

The template generator enables you to create an:

* **External project** (recommended): An isolated project that is not part of the Isaac Lab repository. This approach
  works outside of the core Isaac Lab repository, ensuring that your development efforts remain self-contained. Also,
  it allows your code to be run as an extension in Omniverse.

  .. hint::

    For the external project, the template generator will initialize a new Git repository in the specified directory.
    You can push the generated content to your own remote repository (e.g. GitHub) and share it with others.

* **Internal task**: A task that is part of the Isaac Lab repository. This approach should only be used to create
  new tasks within the Isaac Lab repository in order to contribute to it.

Running the template generator
------------------------------

Install Isaac Lab by following the `installation guide <../../setup/installation/index.html>`_.
We recommend using conda installation as it simplifies calling Python scripts from the terminal.

Then, run the following command to generate a new external project or internal task:

.. tab-set::
  :sync-group: os

  .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

        ./isaaclab.sh --new  # or "./isaaclab.sh -n"

  .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

        isaaclab.bat --new  :: or "isaaclab.bat -n"

The generator will guide you in setting up the project/task for your needs by asking you the following questions:

* Type of project/task (external or internal), and project/task path or names according to the selected type.
* Isaac Lab workflows (see :ref:`feature-workflows`).
* Reinforcement learning libraries (see :ref:`rl-frameworks`), and algorithms (if the selected libraries support multiple algorithms).

External project usage (once generated)
---------------------------------------

Once the external project is generated, a ``README.md`` file will be created in the specified directory.
This file will contain instructions on how to install the project and run the tasks.

Here are some general commands to get started with it:

.. note::

  If Isaac Lab is not installed in a conda environment or in a (virtual) Python environment, use ``FULL_PATH_TO_ISAACLAB/isaaclab.sh -p``
  (or ``FULL_PATH_TO_ISAACLAB\isaaclab.bat -p`` on Windows) instead of ``python`` to run the commands below.

* Install the project (in editable mode).

  .. code:: bash

    python -m pip install -e source/<given-project-name>

* List the tasks available in the project.

  .. warning::

    If the task names change, it may be necessary to update the search pattern ``"Template-"``
    (in the ``scripts/list_envs.py`` file) so that they can be listed.

  .. code:: bash

    python scripts/list_envs.py

* Run a task.

  .. code:: bash

    python scripts/<specific-rl-library>/train.py --task=<Task-Name>

For more details, please follow the instructions in the generated project's ``README.md`` file.

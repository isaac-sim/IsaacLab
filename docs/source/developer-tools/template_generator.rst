.. _own-project:
.. _template-generator:

Build your own project or task
==============================

The template generator bootstraps the package structure and, when requested,
task registration and agent configurations for an Isaac Lab task. Use it to
create either a standalone project outside Isaac Lab or a task intended for
contribution to the Isaac Lab repository. An external project can start with a
working Cartpole example or an empty task package. Internal tasks start from the
Cartpole example.

Choose what to generate
-----------------------

The first prompt chooses where the new task will live.

.. list-table::
   :widths: 22 48 30
   :header-rows: 1

   * - Type
     - Use it when
     - Result
   * - External (recommended)
     - You are creating an application, experiment, or reusable project outside
       the Isaac Lab repository.
     - A standalone, installable uv project using a ``src`` layout.
   * - Internal
     - You intend to contribute the task to the Isaac Lab repository.
     - A task package under ``source/isaaclab_tasks``.

Installed Isaac Lab wheels only offer external projects. The internal option is
available from a source checkout because it writes directly into that checkout.

For an external project, provide one or more authors and choose its initial
content:

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Content
     - Result
   * - Cartpole
     - A runnable example with the selected workflows, task registration, and
       agent configurations.
   * - Blank
     - Package structure, project-owned asset paths, task discovery, tests, and
       development tooling without example task or agent files.

The selected authors are written to ``pyproject.toml``, the project extension
metadata when selected, and the BSD-3-Clause ``LICENSE`` used by Isaac Lab. A
Blank project skips task family, robot, workflow, and RL prompts. A Cartpole
project next asks for one or more task workflows. See :ref:`feature-workflows`
for the complete comparison.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Workflow
     - Good fit
   * - Manager-based | single-agent
     - Most new tasks. Observations, actions, rewards, events, and terminations
       remain modular and easy to replace.
   * - Direct | single-agent
     - Tasks that need custom step and reset control or a compact environment
       implementation.
   * - Direct | multi-agent
     - Tasks with multiple policies or agent-specific observation and action
       spaces.

For Cartpole content, finally choose the RL libraries and algorithms whose
configuration files you want created. The prompt adapts the available choices
to the selected workflow. See :ref:`rl-frameworks` for the framework comparison.

Create and run an external project
----------------------------------

First, :ref:`install Isaac Lab <isaaclab-installation-root>`. Run the generator
from the Isaac Lab source checkout or from a uv project that contains the
installed Isaac Lab package:

.. code-block:: bash

   uv run isaaclab --new

The command uses the dependencies from the active Isaac Lab environment. It
does not invoke ``pip`` or install another set of template dependencies, so it
also works in the pip-less virtual environments created by ``uv``.
When invoked from an installed wheel, the generated ``pyproject.toml`` pins
Isaac Lab and its optional extras to that exact version so ``uv sync`` cannot
silently resolve an older release. When invoked from a source checkout, it
instead records editable paths to that checkout and its workspace packages.
This lets development and release branches work before their version is
published while preserving the code that generated the project.
The source paths are relative to the generated project; regenerate the project
or update ``[tool.uv.sources]`` if either directory moves.

The short form is equivalent:

.. code-block:: bash

   uv run isaaclab -n

Select the following options for a small first project:

* **External** project
* A parent directory outside the Isaac Lab repository
* A Python-compatible project name, such as ``my_robot_project``
* Your name or organization as the author
* **Cartpole** initial content
* A task family name, such as ``balance``
* A robot/config name, such as ``cartpole``
* **No** Isaac Sim UI extension for a headless task package
* **Manager-based | single-agent** workflow
* **rsl_rl** with **PPO**

The prompts display the valid options and accept numbered, comma-separated
selections when more than one choice is allowed. The generator creates the
project under ``<parent-directory>/<project-name>`` and initializes a Git
repository there.

Automate project generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive prompts remain the default. Pass ``--non_interactive`` to opt into
argument-driven generation for scripts and continuous integration. The smallest
external Cartpole command is:

.. code-block:: bash

   uv run isaaclab --new --non_interactive \
      --project_path /work/projects \
      --name my_robot_project \
      --author "Example Author"

This uses the ``cartpole`` content, ``balance`` task family, ``cartpole`` robot
configuration, manager-based single-agent workflow, RSL-RL, and PPO defaults.
Create a Blank project by adding ``--initial_content blank``. Blank projects do
not accept task, workflow, library, or algorithm options because they contain no
example task.

Use the following arguments to override the defaults:

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Argument
     - Meaning
   * - ``--task_type external|internal``
     - Generate an external project by default, or an internal task from a
       source checkout.
   * - ``--project_path PATH``
     - Parent directory for an external project. Required in non-interactive
       mode.
   * - ``--name NAME``
     - External project name or internal task folder name. Required in
       non-interactive mode. ``--project_name`` is an alias.
   * - ``--author NAME``
     - External project author. Required; repeat the argument for multiple
       authors.
   * - ``--initial_content blank|cartpole``
     - External project content. The default is ``cartpole``.
   * - ``--task_name NAME`` and ``--robot_name NAME``
     - Cartpole task family and robot/config names. The defaults are ``balance``
       and ``cartpole``.
   * - ``--include_ui_extension``
     - Include files for loading the project through the Isaac Sim Extension
       Manager.
   * - ``--workflow WORKFLOW``
     - Repeat for any of ``manager-based:single-agent``,
       ``direct:single-agent``, and ``direct:multi-agent``.
   * - ``--rl_library LIBRARY``
     - Repeat for ``rsl_rl``, ``rl_games``, ``skrl``, or ``sb3``.
   * - ``--rl_algorithm ALGORITHM``
     - Repeat an algorithm such as ``ppo``. Prefix it with a selected library,
       such as ``skrl:ippo``, when selecting algorithms independently for
       several libraries.

For example, create direct single-agent and multi-agent variants with selected
SKRL algorithms:

.. code-block:: bash

   uv run isaaclab --new --non_interactive \
      --project_path /work/projects \
      --name multi_workflow_project \
      --author "Example Author" \
      --workflow direct:single-agent \
      --workflow direct:multi-agent \
      --rl_library skrl \
      --rl_algorithm skrl:ppo \
      --rl_algorithm skrl:ippo

Arguments that configure project content are rejected unless
``--non_interactive`` is present, which prevents an accidental argument from
silently changing the interactive flow. Run ``uv run isaaclab --new --help`` to
see the complete command reference.

From a source checkout, an internal task can also be created without prompts:

.. code-block:: bash

   uv run isaaclab --new --non_interactive \
      --task_type internal \
      --name my_internal_task

Enter the generated project and create its environment:

.. code-block:: bash

   cd <parent-directory>/my_robot_project
   uv sync

This default environment includes the selected RL library and the kit-less
Newton backend. It does **not** install Isaac Sim.

For Cartpole content, list the task name and its available presets:

.. code-block:: bash

   uv run isaaclab list_envs --show_presets

A Blank project reports no project tasks until you add and import a Gymnasium
registration under ``src/<project-name>/tasks``.

Copy the task name from the output, then run a quick smoke test:

.. code-block:: bash

   uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16 --viz newton

If the environment launches and the cart moves, the project is ready to edit.
You can then train and play a policy with the same command surface used by
Isaac Lab itself:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task <TASK_NAME>
   uv run isaaclab play --rl_library rsl_rl --task <TASK_NAME> --checkpoint latest --viz newton

Choose optional features and a simulation backend
-------------------------------------------------

The default ``uv sync`` installs the kit-less Newton backend without Isaac Sim.
The generated project's ``[project.optional-dependencies]`` table mirrors every
extra declared by the Isaac Lab package used to create it. This keeps backend,
visualizer, RL, teleoperation, and other feature extras aligned with that Isaac
Lab version. It includes ``rerun``, ``viser``, and ``all`` when the active Isaac
Lab package provides them.

Pass each extra required by a command. For example:

.. code-block:: bash

   uv run --extra isaacsim isaaclab random_agent \
      --task <TASK_NAME> physics=isaacsim_physx

   uv run --extra isaacsim --extra rerun isaaclab play \
      --rl_library skrl --task <TASK_NAME> --viz rerun

Place ``--extra`` before ``isaaclab`` and repeat it when a command needs several
features. The generated ``all`` extra forwards Isaac Lab's own ``all`` extra;
the exact contents remain defined by that release. Inspect the generated
``pyproject.toml`` for the complete version-specific list. See
:ref:`backends-and-presets` for the backend and preset model and
:ref:`isaac-lab-quickstart` for supported physics, renderer, and visualizer
combinations.

.. _project-structure:

Understand the generated project
--------------------------------

An external project is a single installable Python package with the same
standard uv ``src`` layout used by maintained downstream examples. Its root
``pyproject.toml`` declares the package, development tools, backend extras, and
``isaaclab.tasks`` entry point.

The project name identifies the repository and Python package. Task-wide MDP
terms live under the separately named task family. Robot-specific scenes,
registrations, and agent configurations live under ``config/<robot-name>``.
This separation lets a project add another robot configuration without copying
the task MDP, or add another task family without creating another repository.

Project-owned USD files and related data live under
``src/<project-name>/assets/data``. The project's ``assets`` module exposes a
package-relative ``<PROJECT_NAME>_ASSETS_DIR`` path, so task and asset
configurations can locate those files from an editable checkout or an installed
wheel. The generated ``.gitattributes`` routes USD files through Git LFS.

Code shared by several task families can live in a package such as
``tasks/mdp``. The generated task importer skips packages named ``mdp`` while it
searches for task registrations, so shared modules do not register as task
families.

A Cartpole project resembles:

.. code-block:: text

   my_robot_project/
   ├── LICENSE
   ├── pyproject.toml
   ├── README.md
   ├── scripts/
   │   └── list_envs.py
   ├── src/
   │   └── my_robot_project/
   │       ├── __init__.py
   │       ├── assets/
   │       │   ├── __init__.py
   │       │   └── data/
   │       └── tasks/
   │           ├── __init__.py
   │           └── balance/
   │               ├── mdp/
   │               └── config/
   │                   └── cartpole/
   │                       ├── agents/
   │                       └── env_cfg.py
   └── tests/
       └── test_registration.py

The generated package ``__init__.py`` is intentionally passive. Installing the
project exposes ``my_robot_project.tasks`` through the ``isaaclab.tasks`` entry
point, so importing the package for utilities does not eagerly register tasks.
For Blank content, the ``tasks`` directory contains only ``__init__.py`` until
you add and import a task package.

If you opt into the Isaac Sim UI extension, the generator additionally creates
``config/extension.toml`` and ``src/my_robot_project/ui_extension_example.py``.
Launch Isaac Sim with the generated ``isaacsim`` extra when using it. The
default is a headless task package and does not include these files.

Run commands from the project root so ``uv`` can find the package and task entry
point. Commit ``pyproject.toml`` and ``uv.lock`` to give collaborators the same
dependency resolution. For a project linked to a source checkout, collaborators
must also place that checkout at the recorded relative path or update the source
entries.

Develop the generated task
--------------------------

Start with a dummy agent before training. A zero-action agent is useful for
checking resets and passive dynamics, while a random-action agent also exercises
the action and observation paths:

.. code-block:: bash

   uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
   uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

Both commands run until their step limit, their visualizer closes, or you press
Ctrl+C. Ctrl+C closes the environment and exits without a traceback.

Edit the generated environment configuration and task terms under
``src/<project-name>/tasks``. The generated package is
installed in editable mode, so you do not need to reinstall it after each
change.

Use the remaining project commands as the task matures:

.. code-block:: bash

   uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> \
      --task <TASK_NAME> --num_gpus 2
   uv run isaaclab benchmark runtime --task <TASK_NAME> \
      --num_envs 16 --num_steps 1000
   uv run pre-commit run --all-files

The generator includes ``tests/test_registration.py`` to verify the task IDs,
environment entry points, and default agent. Its ``pyproject.toml`` installs
pytest for development and registers the ``unit``, ``integration``, ``smoke``,
and ``kitless`` markers. Add project-owned behavioral tests under ``tests`` and
run them with:

.. code-block:: bash

   uv run pytest tests

The reusable-looking helpers under ``source/isaaclab_tasks/test`` belong to the
Isaac Lab repository test suite and are not installed with ``isaaclab_tasks``.
External projects should build their environment harness from public APIs and
maintain project-local fixtures. Copying ``env_test_utils.py`` into a project is
vendoring it, so the project must track upstream changes to that copy.

To configure VS Code or Cursor, run the generated setup task or invoke it directly:

.. code-block:: bash

   uv run isaaclab --editor

The command selects the active interpreter and creates a git-ignored
``pyrightconfig.json``. This child configuration inherits the checked-in
Pyright policy from ``pyproject.toml`` and adds the generated project's
``src`` import root, installed Isaac Lab packages, and any discovered Isaac
Sim extensions. When using the ``isaacsim`` extra, include it while generating
the configuration:

.. code-block:: bash

   uv run --extra isaacsim isaaclab --editor

In VS Code, use Pylance and select the interpreter that ran the setup command.
In Cursor, install the ``detachhead.basedpyright`` extension instead of Pylance,
select the same interpreter, and reload the window. Both language servers read
the generated ``pyrightconfig.json``.

Create an internal task
-----------------------

Choose **Internal** only when working from an Isaac Lab source checkout. The
generator writes the new task into ``source/isaaclab_tasks`` instead of creating
a separate project. From the Isaac Lab repository root, list and test it with:

.. code-block:: bash

   uv run isaaclab list_envs --show_presets
   uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16
   uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>

The training command automatically selects an agent configuration when the
generated task has only one entry point for that RL library. If you generated
multiple algorithms, or want to select one explicitly, pass its registered
entry-point name. Non-PPO entry points include the algorithm name; for example:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task <TASK_NAME> \
      --agent rsl_rl_distillation_cfg_entry_point

Troubleshooting
---------------

**The project path is rejected**
   External projects must live outside the Isaac Lab repository. Enter the
   parent directory; the generator appends the project name automatically.

**The project name is rejected**
   Use a valid Python identifier containing letters, numbers, and underscores,
   without spaces or hyphens. The name cannot begin with a number.

**The task family or robot/config name is rejected**
   Use a valid Python identifier for each name. These names become package
   directories under ``src/<project-name>/tasks``.

**The CLI cannot find the generated task**
   Run ``uv sync`` and invoke the command from the generated project root. Then
   confirm the task appears in ``uv run isaaclab list_envs``. Pass ``--all`` to
   include tasks from every installed task package instead of limiting the list
   to the current project's ``isaaclab.tasks`` entry point.

**Non-interactive arguments are rejected**
   Include ``--non_interactive`` after ``--new``. Cartpole is the default initial
   content; Blank content rejects task, workflow, library, and algorithm options.
   The error message identifies missing required arguments or incompatible
   selections.

**An optional backend module is missing**
   Add its extra before the command, such as ``uv run --extra ovphysx
   isaaclab ...`` or ``uv run --extra isaacsim --extra rerun isaaclab ...``.
   Every extra offered by the active Isaac Lab package is copied into the
   generated project's optional dependencies.

**The generator reports a missing template dependency**
   Current versions obtain the renderer and prompts from the Isaac Lab
   environment; no manual ``pip install`` is required. Update the Isaac Lab
   checkout or installed package and run the generator again.

The generated ``README.md`` contains the same project-local commands and should
be kept up to date as the project evolves.

Contribution Guidelines
=======================

.. seealso::

   This page is the source of truth for the ``isaaclab-following-coding-style``,
   ``isaaclab-preparing-pr-workflow``, and ``isaaclab-writing-changelog-fragments`` agent skills
   (`skills/developer/coding-style/ <../../../skills/developer/coding-style/SKILL.md>`__,
   `skills/developer/pr-workflow/ <../../../skills/developer/pr-workflow/SKILL.md>`__,
   `skills/developer/changelog-fragments/ <../../../skills/developer/changelog-fragments/SKILL.md>`__).
   When shared guidance changes, update affected skill workflows and examples without copying the rules. See
   :doc:`/source/developer-tools/agent_skills`.

We wholeheartedly welcome contributions to the project to make the framework more mature
and useful for everyone. These may happen in forms of:

* Bug reports: Please report any bugs you find in the `issue tracker <https://github.com/isaac-sim/IsaacLab/issues>`__.
* Feature requests: Please suggest new features you would like to see in the `discussions <https://github.com/isaac-sim/IsaacLab/discussions>`__.
* Code contributions: Please submit a `pull request <https://github.com/isaac-sim/IsaacLab/pulls>`__.

  * Bug fixes
  * New features
  * Documentation improvements
  * Tutorials and tutorial improvements

We prefer GitHub `discussions <https://github.com/isaac-sim/IsaacLab/discussions>`_ for discussing ideas,
asking questions, conversations and requests for new features.

Please use the
`issue tracker <https://github.com/isaac-sim/IsaacLab/issues>`_ only to track executable pieces of work
with a definite scope and a clear deliverable. These can be fixing bugs, new features, or general updates.


Contributing Code
-----------------

.. attention::

   Please refer to the `Google Style Guide <https://google.github.io/styleguide/pyguide.html>`__
   for the coding style before contributing to the codebase. In the coding style section,
   we outline the specific deviations from the style guide that we follow in the codebase.

We use `GitHub <https://github.com/isaac-sim/IsaacLab>`__ for code hosting. Please
follow the following steps to contribute code:

1. Create an issue in the `issue tracker <https://github.com/isaac-sim/IsaacLab/issues>`__ to discuss
   the changes or additions you would like to make. This helps us to avoid duplicate work and to make
   sure that the changes are aligned with the roadmap of the project.
2. Fork the repository.
3. Create a new branch for your changes.
4. Make your changes and commit them.
5. Push your changes to your fork.
6. Submit a pull request to the `develop branch <https://github.com/isaac-sim/IsaacLab/compare/develop...>`__.
7. Ensure all the checks on the pull request template are performed.

After sending a pull request, the maintainers will review your code and provide feedback.

Ensure that your code is formatted and documented, and run the relevant tests and required CI checks
as described in `Unit Testing`_ and `Tools`_.

.. tip::

   It is important to keep the pull request as small as possible. This makes it easier for the
   maintainers to review your code. If you are making multiple changes, please send multiple pull requests.
   Large pull requests are difficult to review and may take a long time to merge.


More details on the code style and testing can be found in the `Coding Style`_ and `Unit Testing`_ sections.


Contributing Documentation
--------------------------

Contributing to the documentation is as easy as contributing to the codebase. All the source files
for the documentation are located in the ``IsaacLab/docs`` directory. The documentation is written in
`reStructuredText <https://docutils.sourceforge.io/rst.html>`__ format.

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`__ with the
`Book Theme <https://sphinx-book-theme.readthedocs.io/en/stable/>`__
for maintaining the documentation.

Sending a pull request for the documentation is the same as sending a pull request for the codebase.
Please follow the steps mentioned in the `Contributing Code`_ section.

For documentation media, only small ``.jpg`` files should be committed directly to the Isaac Lab repository.
Upload larger images, videos, animations, and other media to an external hosting location, then link to or embed
the externally hosted file from the documentation.

.. caution::

  Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`__ before building
  the documentation. The build command creates a temporary environment for the
  ``dev`` extra, which includes documentation requirements, leaving the
  repository's ``.venv`` unchanged.


To build the documentation, run the following command from the repository root. It installs
the documentation packages and builds the current version:

.. code:: bash

   uv run isaaclab --docs

The documentation is generated in the ``docs/_build`` directory. To view the documentation, open
the ``index.html`` file in ``docs/_build/current``. This can be done by running the following command
in the terminal:

.. code:: bash

   xdg-open docs/_build/current/index.html

.. hint::

   The ``xdg-open`` command is used to open the ``index.html`` file in the default browser. If you are
   using a different operating system, you can use the appropriate command to open the file in the browser.


For PR validation, remove the generated HTML before building so deleted pages cannot leave stale output.
Run these commands from the repository root; they preserve the Sphinx cache:

.. code:: bash

   uv run python -c "import shutil; shutil.rmtree('docs/_build/current', ignore_errors=True)"
   uv run isaaclab --docs


Contributing assets
-------------------

Large asset files should not be added directly to the Isaac Lab repository. Instead, host them in a
separate repository and link to them from the relevant documentation.

Please checkout the `Isaac Sim Assets <https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_overview.html>`__
for more information on what is presently available.

.. attention::

  We are currently working on a better way to contribute assets. We will update this section once we
  have a solution. In the meantime, please follow the steps mentioned below.

To host your own assets, the current solution is:

1. Create a separate repository for the assets and add it over there
2. Make sure the assets are licensed for use and distribution
3. Include images of the assets in the README file of the repository
4. Send a pull request with a link to the repository

We will then verify the assets and their licensing and determine how to integrate them. If you have
questions, please open an issue in the repository.


Maintaining package changelogs and versions
-------------------------------------------

Each release-managed package maintains a changelog in ``docs/CHANGELOG.rst`` and its version in
``pyproject.toml``.

The changelog contains the curated, chronologically ordered list of notable changes for each
package version.

.. note::

   ``CHANGELOG.rst`` and the package version in ``pyproject.toml`` are compiled by CI from per-PR **fragment
   files** — contributors do not edit them directly. For every package your PR touches
   in ``source/<pkg>/`` (outside ``changelog.d/``), add one fragment under
   ``source/<pkg>/changelog.d/<slug>.<tier>.rst``:

   * ``<slug>.rst`` — patch bump
   * ``<slug>.minor.rst`` — minor bump (new public API)
   * ``<slug>.major.rst`` — major bump (breaking change)
   * ``<slug>.skip`` — no entry, no bump (CI / docs / test-only PRs)

   ``<slug>`` is any short, unique name; your branch name with ``/`` replaced by ``-``
   is the recommended default. Within a batch the highest tier wins for the package.
   The package version in ``pyproject.toml`` is bumped by CI according to
   `Semantic Versioning <https://semver.org/>`__.

The changelog file is written in `reStructuredText <https://docutils.sourceforge.io/rst.html>`__ format.
The goal of this changelog is to help users and contributors see precisely what notable changes have
been made between each release of a package. This is a *MUST* for every release-managed package.

For each fragment, please follow the following guidelines:

* Each fragment is divided into subsections based on the type of changes made.

  * ``Added``: For new features.
  * ``Changed``: For changes in existing functionality.
  * ``Deprecated``: For soon-to-be removed features.
  * ``Removed``: For now removed features.
  * ``Fixed``: For any bug fixes.

* Each change is described in its corresponding sub-section with a bullet point.
* Prefix breaking changes with **Breaking:** and provide migration guidance for deprecated, changed,
  or removed behavior that requires callers to adapt.
* The bullet points are written in the **past tense**.

  * This means that the change is described as if it has already happened.
  * The bullet points should be concise and to the point. They should not be verbose.
  * The bullet point should also include the reason for the change, if applicable.


.. tip::

   When in doubt, please check the style in the existing changelog files and follow the same style.

For example, ``source/isaaclab/changelog.d/fix-partial-reset.rst``:

.. code:: rst

    Fixed
    ^^^^^

    * Fixed contact sensor reset behavior when only a subset of environments was reset.


Coding Style
------------

We follow the `Google Style
Guides <https://google.github.io/styleguide/pyguide.html>`__ for the
codebase. For Python code, the PEP guidelines are followed. Most
important ones are `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`__
for code comments and layout,
`PEP-484 <http://www.python.org/dev/peps/pep-0484>`__ and
`PEP-585 <https://www.python.org/dev/peps/pep-0585/>`__ for
type-hinting.

For documentation, we adopt the `Google Style Guide <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__
for docstrings. We use `Sphinx <https://www.sphinx-doc.org/en/master/>`__ for generating the documentation.
Please make sure that your code is well-documented and follows the guidelines.

Refactoring and API Design
^^^^^^^^^^^^^^^^^^^^^^^^^^

Make the smallest change that solves the problem. Read surrounding code, callers, tests, and documentation
before changing an interface. Apply these rules when adding code or cleaning up existing implementations:

* Preserve observable behavior unless the change explicitly requires otherwise. Check ordering, shapes,
  dtype, device, input mutation, and failure behavior as well as return values. Public API removals and
  renames require a deprecation and migration path.
* Prefer a functional approach for stateless computations and transformations: use plain functions with
  explicit inputs and outputs, keeping side effects at clear boundaries. Use classes when they own
  meaningful state or resources, enforce invariants over a lifecycle, or implement an interface required
  by the architecture. Avoid classes that only group static methods, wrap a single operation, or forward
  calls to another object. Preserve established public contracts when simplifying existing designs.
* Reuse existing mechanisms before introducing helpers, configuration options, or abstractions. Extract
  shared logic when it has the same contract; keep helpers private unless callers need a public API.
  Prefer direct control flow and early returns when they remove unnecessary nesting.
* Inline simple expressions and operations when a helper would only add indirection. Do not extract
  a one-line helper merely to rename an obvious operation. Introduce a helper when it removes meaningful
  duplication or gives a coherent, non-trivial operation a useful name; its benefit should outweigh the
  need to jump to another definition to understand the caller.
* Prefer direct attribute access and assignment (``obj.value`` and ``obj.value = value``). Use ``getattr``
  and ``setattr`` only when dynamic attribute access is required, such as when the attribute name is
  determined at runtime. Do not use them for known attributes or use default values to hide a missing
  required attribute; express optional fields explicitly in the interface.
* Give each piece of state and validation one owner. Consumers should use the owner's contract instead
  of repairing results or maintaining duplicate state. Cache derived values only when their lifetime and
  invalidation are clear; do not expose mutable cached results for callers to modify accidentally.
* Keep backend selection at shared dispatch boundaries. Use established types, configuration, and
  capability contracts instead of inferring behavior from class-name strings.
* Keep physics and rendering responsibilities separate and resolve construction requirements before
  finalization. See :doc:`/source/developer-tools/scene_data_providers` for geometry ownership and
  :doc:`/source/developer-tools/add_physics_backend` for backend integration.
* Prefer existing project dependencies and the standard library. Do not add dependencies or compatibility
  layers for hypothetical future uses.

Array Operations and Runtime Cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Review allocations, copies, synchronization, and recomputation in code that runs per step, reset, or
environment. Costs that are small for one environment can dominate a large batch.

* Preserve slices through APIs that support them. Materialize indices only at a consumer that requires
  them, reusing cached device indices when available. Preserve the selector's ordering and device contract.
* Allocate arrays directly with the required value, dtype, and device. Prefer ``torch.full`` or ``wp.full``
  over filling through Python lists, arithmetic on temporary arrays, or a round trip through another library.
* Remove redundant copies and ``contiguous()`` calls only after checking layout and ownership requirements.
  Do not mutate caller-owned inputs unless the API explicitly promises an in-place operation.
* Batch operations when supported. Avoid Python loops over environments and unnecessary host/device
  transfers or scalar reads that synchronize the device in hot paths.
* Allocate expensive optional buffers on first use when their lifecycle permits it. Account for graph
  capture: any required allocation must happen before capture when the allocator requires it.

Naming and Documentation
^^^^^^^^^^^^^^^^^^^^^^^^

* Use ``snake_case`` for functions, methods, and CLI arguments. Keep related public symbols discoverable
  through consistent prefixes and use existing API vocabulary.
* Use concrete types where practical, built-in collection annotations such as ``list[str]``, and
  ``X | None`` for optional values. Keep argument and return types in signatures, without repeating them
  in docstrings.
* Document public APIs with Google-style docstrings. State physical units inline, for example
  ``Particle positions [m], shape [N, 3]``. Use ``[m or rad, depending on joint type]`` for mixed joint
  quantities. Document coordinate frames and array shapes where relevant; indices, counts, and flags
  do not need physical units.
* Keep comments brief and selective. Explain intent, non-obvious constraints, or edge cases that the code
  alone cannot make clear. Avoid narrating implementation steps or repeating what an expression does.
  Readability comments, section markers, and headings that help organize a file are welcome.
* Describe the current contract in code comments. Put migration instructions and descriptions of old
  behavior in public migration documentation or changelog entries, rather than leaving a history of
  refactoring in the implementation. Retain historical context only when it explains a constraint that
  still affects correctness.
* Update public documentation with API changes and verify technical claims against the implementation.

Code Structure
^^^^^^^^^^^^^^

We follow a specific structure for the codebase. This helps in maintaining the codebase and makes it easier to
understand.

Keep short expressions on one line within the configured limit and break longer expressions at
meaningful boundaries. Keep loop headers focused on iteration; unpack bulky nested records in the body.
Prefer descriptive names to new acronyms. Reuse matching sequences or mappings with ``*``/``**`` instead
of unpacking and rebuilding them; do not introduce packing containers or reflective assignment just
to shorten code.

In a Python file, we follow the following structure:

.. code:: python

   # Imports: These are sorted by the pre-commit hooks.
   # Constants
   # Functions (public)
   # Classes (public)
   # _Functions (private)
   # _Classes (private)

Imports are sorted by Ruff through ``uv run isaaclab -f``. The groups are ``__future__``, standard library,
third-party packages, Omniverse runtime packages, Isaac Lab packages, and local relative imports; the
exact package groups are configured in ``pyproject.toml``. Let the formatter apply this order.
Use relative imports within the same package when the target is at most three leading dots away
(for example, ``from ...utils import math as math_utils``). Use absolute imports for deeper targets,
other packages, and modules that run as scripts with an ``if __name__ == "__main__":`` block.

Prefer module-level imports. A local import is appropriate when it defers an optional backend or simulator
dependency until the selected runtime path needs it. Keep configuration imports usable before simulator
startup and avoid repeating runtime initialization already owned by the package. To deal with circular imports, use the
:obj:`typing.TYPE_CHECKING` variable. Please refer to the `Circular Imports`_ section for more details.

Public export modules in ``__init__.py`` are an exception to the above: they use
:func:`~isaaclab.utils.module.lazy_export` instead of traditional imports.
See the `Lazy Loading & Module Exports`_ section for details.

Pass ``ProxyArray`` objects directly to Warp kernels. Keep one proxy per owned array, without
parallel ``_ta``, ``_warp``, or ``_torch`` attributes; timestamped array caches can own the proxy in
``data``. Use explicit native access only where the receiving API requires it.

Python does not have a concept of private and public classes and functions. However, we follow the
convention of prefixing the private functions and classes with an underscore.
The public functions and classes are the ones that are intended to be used by the users. The private
functions and classes are the ones that are intended to be used internally in that file.
Irrespective of the public or private nature of the functions and classes, we follow the Style Guide
for the code and make sure that the code and documentation are consistent.

When a class is warranted, order its members as follows:

.. code:: python

   # Constants
   # Class variables (public or private): Must have the type hint ClassVar[type]
   # Dunder methods, when needed: __init__, __del__
   # Representation: __repr__, __str__
   # Properties: @property
   # Instance methods (public)
   # Class methods (public)
   # Static methods (public)
   # _Instance methods (private)
   # _Class methods (private)
   # _Static methods (private)

The rule of thumb is that the functions within the classes are ordered in the way a user would
expect to use them. For instance, if the class contains the method :meth:`initialize`, :meth:`reset`,
:meth:`update`, and :meth:`close`, then they should be listed in the order of their usage.
The same applies for private functions in the class. Their order is based on the order of call inside the
class.

Include only the members a class needs; this ordering is not a checklist of methods to implement.
For classes that own resources, expose explicit cleanup through ``close()`` or a context manager.

.. dropdown:: Minimal function example
   :icon: code

   .. literalinclude:: snippets/code_skeleton.py
      :language: python

Circular Imports
^^^^^^^^^^^^^^^^

Circular imports happen when two modules import each other, which is a common issue in Python.
You can prevent circular imports by adhering to the best practices outlined in this
`StackOverflow post <https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python>`__.

In general, it is essential to avoid circular imports as they can lead to unpredictable behavior.

However, in our codebase, we encounter circular imports at a sub-package level. This situation arises
due to our specific code structure. We organize classes or functions and their corresponding configuration
objects into separate files. This separation enhances code readability and maintainability. Nevertheless,
it can result in circular imports because, in many configuration objects, we specify classes or functions
as default values using the attributes ``class_type`` and ``func`` respectively.

To address this, we use two complementary techniques:

1. **Resolvable strings** — Store ``class_type`` and ``func`` as ``{DIR}``-based strings
   (e.g. ``"{DIR}.sensor:Sensor"``) so the implementation module is never imported at config
   construction time. The string is resolved to the actual class (via :class:`~isaaclab.utils.string.ResolvableString`)
   on invocation or attribute access that needs the implementation. Initialize any required runtime
   before triggering resolution.
2. **TYPE_CHECKING guards** — Import the implementation class under `typing.TYPE_CHECKING
   <https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING>`_ so that IDEs and type
   checkers can provide autocomplete on the type annotation without triggering a runtime import.

See the `Resolvable Strings`_ and `Lazy Loading & Module Exports`_ sections for full
examples of both patterns.

Configuration Operations
^^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`~isaaclab.utils.instantiate` to construct the implementation selected by a config:

.. code-block:: python

   from isaaclab.utils import clone, instantiate, replace, to_dict, update_from_dict, validate

   robot_cfg = replace(ROBOT_CFG, prim_path="{ENV_REGEX_NS}/Robot")
   other_cfg = clone(robot_cfg)
   update_from_dict(other_cfg, {"init_state": {"pos": (1.0, 0.0, 0.0)}})
   validate(other_cfg)
   settings = to_dict(other_cfg)
   robot = instantiate(robot_cfg)
   action = instantiate(action_cfg, env)

``instantiate`` passes the config as the first constructor argument, followed by any additional
arguments. It does not copy configs, construct nested configs, or cache instances. Resource
sharing remains the responsibility of ``SimulationContext.get_or_create_backend``.

``clone`` and ``replace`` return new configurations, preserving fields explicitly marked as borrowed.
``update_from_dict`` updates an existing configuration in place. ``validate`` checks required
fields and runs nested ``validate_config`` hooks. Prefer these functions in new code; the existing
``cfg.copy()``, ``cfg.replace(...)``, ``cfg.validate()``, ``cfg.to_dict()``, ``cfg.from_dict(...)``, and
``cfg.class_type(cfg, ...)`` calls remain supported without deprecation.
The longer function names ``class_to_dict`` and ``update_class_from_dict`` also remain supported.

Lazy Loading & Module Exports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use **lazy loading** for public package exports so that importing a top-level package
(e.g. ``import isaaclab.sensors``) does not eagerly pull in heavyweight dependencies like
``pxr``, ``omni``, or ``scipy``. This is critical because config classes must be constructable
*before* ``SimulationApp`` is launched.

We follow `SPEC 1 — Lazy Loading of Submodules and Functions
<https://scientific-python.org/specs/spec-0001/>`__ and use the `lazy_loader
<https://pypi.org/project/lazy-loader/>`__ library (endorsed by NumPy, SciPy, scikit-image,
scikit-learn, NetworkX) with ``.pyi`` type-stub files. The stub is the **single source of
truth** for both IDE autocomplete and runtime lazy loading.

**Standard pattern for public export modules:**

.. code:: python

   # mypackage/__init__.py
   from isaaclab.utils.module import lazy_export

   lazy_export()

With a corresponding type stub adjacent to it:

.. code:: python

   # mypackage/__init__.pyi
   __all__ = ["MyClass", "MyOtherClass", "my_function"]

   from .my_module import MyClass, MyOtherClass
   from .my_other_module import my_function

Key rules for ``.pyi`` stubs:

* The ``__all__`` list at the top marks names as public re-exports (per `PEP 484
  <https://peps.python.org/pep-0484/#stub-files>`__).
* Group imports from the same submodule on one line. Use parenthesized multi-line
  imports if the line exceeds 100 characters.
* Use **relative imports** (``from .something import ...``) for local submodule
  symbols. Absolute wildcard imports (``from pkg import *``) are only used for
  cross-package fallbacks (see below).
* Include the standard Isaac Lab license header.

**Cross-package fallback** — for modules that re-export names from another package
(e.g. task MDP modules that delegate to ``isaaclab.envs.mdp``), add a wildcard
import for the external package in the ``.pyi`` stub:

.. code:: python

   # isaaclab_tasks/.../mdp/__init__.pyi
   __all__ = ["MyReward", "MyObservation"]

   from .rewards import MyReward
   from .observations import MyObservation

   from isaaclab.envs.mdp import *

The ``__init__.py`` stays the same as the standard pattern — just ``lazy_export()``
with no arguments:

.. code:: python

   # isaaclab_tasks/.../mdp/__init__.py
   from isaaclab.utils.module import lazy_export

   lazy_export()

At runtime, ``lazy_export`` parses the ``.pyi`` stub and uses the absolute wildcard
import (``from isaaclab.envs.mdp import *``) as a fallback: any name not found in
the local submodules is looked up in the specified package. This also gives type
checkers and IDEs full visibility into the re-exported symbols.

**Relative wildcard re-exports** — the stub can also use ``from .submodule import *``
to eagerly export all public names from a local submodule. This is resolved at
import time (not lazily). A large or frequently changing API alone does not justify eager imports.

.. note::

   Relative wildcard re-exports bypass lazy loading and eagerly import every public
   name from the submodule at package init time. In general, we advise against using
   them unless absolutely necessary. Prefer listing explicit named imports in the stub
   so that the public API surface is clear, reviewable, and remains lazily loaded.

.. code:: python

   # isaaclab_tasks/.../mdp/__init__.pyi
   from .rewards import *
   from .observations import *

   from isaaclab.envs.mdp import *

**Ensuring .pyi stubs are distributed**

Declare stub files in the package's ``pyproject.toml`` so they are included in distributions:

.. code:: toml

   [tool.setuptools.package-data]
   "*" = ["*.pyi"]

Keep this configuration in packages that provide lazy export stubs. The pre-commit ``insert-license`` hook
is configured to add license headers to ``.pyi`` files automatically (``\.(pyi?|ya?ml)$``).

Resolvable Strings
^^^^^^^^^^^^^^^^^^

When a config field needs to reference a class or callable that depends on the simulator
runtime, store it as a :class:`~isaaclab.utils.string.ResolvableString` rather than a
direct reference. This avoids eagerly importing heavyweight modules (``omni``, ``pxr``,
etc.) at config construction time. Invocation or attribute access that needs the implementation triggers
resolution; the caller must initialize any required runtime first. Resolution does not itself wait for
``SimulationApp``, and workflows without Kit need not launch it.

You can use either the ``{DIR}`` shorthand or a fully-qualified module path:

.. code:: python

   # Good — {DIR} shorthand (resolved to the current package at runtime)
   class_type: type[Sensor] | str = "{DIR}.sensor:Sensor"

   # Good — fully-qualified path (useful for cross-package references)
   class_type: type[Sensor] | str = "isaaclab.sensors.my_sensor.sensor:Sensor"

   # Bad — eagerly imports the implementation module
   from .sensor import Sensor
   class_type: type = Sensor

The config machinery expands ``{DIR}`` to the package of the field's defining class
(e.g. ``isaaclab.sensors.my_sensor``), without importing the referenced implementation. Prefer ``{DIR}``
for references within the same package since it stays correct across renames and moves.

For the type annotation (``type[Sensor]``), import the class under a ``TYPE_CHECKING`` guard
so that the IDE can still provide autocomplete without triggering a runtime import:

.. code:: python

   from __future__ import annotations
   import typing

   if typing.TYPE_CHECKING:
       from .sensor import Sensor

Config + Implementation File Split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For components with configuration classes, keep the configuration and runtime implementation in
separate files so importing configuration does not load heavy runtime dependencies. This pattern does
not require introducing a class or configuration object for a stateless function:

.. code:: text

   my_sensor/
   ├── __init__.py          # lazy_export()
   ├── __init__.pyi         # re-exports: SensorCfg, Sensor
   ├── sensor_cfg.py        # pure data — no runtime deps
   └── sensor.py            # implementation — may import omni, pxr, etc.

``__init__.py`` — uses ``lazy_export()`` to lazily load names from the stub:

.. code:: python

   # my_sensor/__init__.py
   from isaaclab.utils.module import lazy_export

   lazy_export()

``__init__.pyi`` — declares the public API for both IDE autocomplete and lazy loading:

.. code:: python

   # my_sensor/__init__.pyi
   __all__ = ["SensorCfg", "Sensor"]

   from .sensor_cfg import SensorCfg
   from .sensor import Sensor

``sensor_cfg.py`` — pure data; references the implementation class by resolvable string
to avoid importing it:

.. code:: python

   # my_sensor/sensor_cfg.py
   from __future__ import annotations
   import typing

   from isaaclab.utils import configclass

   if typing.TYPE_CHECKING:
       from .sensor import Sensor

   @configclass
   class SensorCfg:
       class_type: type[Sensor] | str = "{DIR}.sensor:Sensor"

``sensor.py`` — the implementation; imports runtime dependencies only when needed:

.. code:: python

   # my_sensor/sensor.py
   from pxr import Usd

   from .sensor_cfg import SensorCfg

   class Sensor:
       def __init__(self, cfg: SensorCfg, stage: Usd.Stage) -> None:
           self.cfg = cfg
           self.stage = stage

Type-hinting
^^^^^^^^^^^^

Use specific type hints in function signatures and class attributes. Describe meaning, units, shapes,
and constraints in docstrings without repeating the annotated types:

.. code:: python

   def add(a: int, b: int) -> int:
       """Add two integers.

       Args:
           a: The first operand.
           b: The second operand.

       Returns:
           The sum of the operands.
       """
       return a + b

* Prefer built-in collection types such as ``list[str]`` and ``dict[str, int]``.
* Use ``X | None`` for optional values.
* Use ``TYPE_CHECKING`` and deferred annotations when types require runtime-only imports.

* Annotate functions that return no value with ``-> None``. Omit an unnecessary ``Returns:`` section
  from their docstrings.

Documenting the code
^^^^^^^^^^^^^^^^^^^^

Write documentation that lets a caller use the API without reading its implementation.

* State the behavior first, then explain constraints, defaults, and relevant failure modes.
* Document units, coordinate frames, shapes, and ownership or mutation of inputs and returned data.
* Explain non-obvious design constraints in brief comments near the relevant code.
* Use a small example when it clarifies usage. Avoid repeating the signature or narrating each operation.
* Use plain language and active voice; remove repetition and vague qualifiers.
* Update documentation when behavior changes and check examples against the current implementation.


Unit Testing
------------

We use `pytest <https://docs.pytest.org>`__ for unit testing. Keep coverage lean and fast by giving each
contract one primary test owner at the strongest observable boundary. Start with existing coverage at
that boundary and extend it when it can clearly cover the changed behavior.

Apply the authoring gate and retention criteria in the
`test-audit skill <../../../skills/developer/test-audit/SKILL.md>`__ when adding, changing, reviewing,
or pruning tests. It maintains the detailed criteria for deciding whether a test earns its cost.

* Add a test only for a distinct behavior, regression, boundary, or failure mode that existing coverage
  does not already exercise. Formatting and mechanical cleanup do not automatically require new tests.
  Before adding it, identify the protected contract, a credible regression, and why existing coverage
  would miss that regression. Do not add production exports, flags, wrappers, or injection hooks solely
  to support a test; exercise the real boundary instead.
* Test observable behavior and public contracts. Avoid assertions tied to private implementation details
  or expected values computed by repeating the production algorithm. Avoid assertion-free smoke tests,
  self-comparisons, and mocks or fixtures that supply the very behavior the test claims to verify.
* Verify that regression tests fail without the fix for the intended reason and pass with it. Cover the
  bug at its owning boundary; another layer or backend needs a distinct risk to justify replaying it.
* Keep parameter matrices and simulation fixtures focused on distinct execution paths. Consolidate
  redundant coverage instead of adding overlapping cases or rebuilding the same scene unnecessarily.
  Avoid Cartesian products of devices, shapes, and environment counts when the axes do not exercise
  distinct paths. Reuse the fixture that already establishes the contract.
* Before removing or merging coverage, identify the proof that remains and demonstrate that it fails
  when the contract is broken, using a focused mutation of the production owner where appropriate.
  Preserve distinct edge cases and independent API, physical, backend-parity, and packaging contracts.
  A slow test or similar-looking assertions alone are not evidence of duplication.
* Run the narrowest relevant test first. If an optional dependency is missing, identify its project extra
  and retry with ``uv run --extra <extra> python -m pytest ...``.
  For changes intended to reduce test time, measure before and after on the same machine and separate
  kernel compilation from execution time.

Use the same commands on Linux and Windows:

.. code-block:: bash

   # Run a particular test
   uv run python -m pytest source/isaaclab/test/utils/test_circular_buffer.py::test_reset

   # Run all tests in a particular file
   uv run python -m pytest source/isaaclab/test/utils/test_circular_buffer.py

   # Run source-package tests through the repository test runner
   uv run python tools/run_all_tests.py

   # Run tooling tests under tools/
   uv run isaaclab --test

All of these commands exit with a nonzero code when tests fail, so a test
failure fails the invoking shell or CI step as well.

Tools
-----

We use the following tools for maintaining code quality:

* `pre-commit <https://pre-commit.com/>`__: Runs a list of formatters and linters over the codebase.
* `ruff <https://github.com/astral-sh/ruff/>`__: An extremely fast Python linter and formatter.

Run the repository formatting and lint checks from the uv-managed environment on Linux or Windows:

.. code-block:: bash

   uv run isaaclab --format

Contribution Guidelines
=======================

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
6. Submit a pull request to the `main branch <https://github.com/isaac-sim/IsaacLab/compare>`__.
7. Ensure all the checks on the pull request template are performed.

After sending a pull request, the maintainers will review your code and provide feedback.

Please ensure that your code is well-formatted, documented and passes all the tests.

.. tip::

   It is important to keep the pull request as small as possible. This makes it easier for the
   maintainers to review your code. If you are making multiple changes, please send multiple pull requests.
   Large pull requests are difficult to review and may take a long time to merge.


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

.. caution::

  To build the documentation, we recommend creating a `virtual environment <https://docs.python.org/3/library/venv.html>`__
  to install the dependencies. This can also be a `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.


To build the documentation, run the following command in the terminal which installs the required python packages and
builds the documentation using the ``docs/Makefile``:

.. code:: bash

   ./isaaclab.sh --docs  # or "./isaaclab.sh -d"

The documentation is generated in the ``docs/_build`` directory. To view the documentation, open
the ``index.html`` file in the ``html`` directory. This can be done by running the following command
in the terminal:

.. code:: bash

   xdg-open docs/_build/current/index.html

.. hint::

   The ``xdg-open`` command is used to open the ``index.html`` file in the default browser. If you are
   using a different operating system, you can use the appropriate command to open the file in the browser.


To do a clean build, run the following command in the terminal:

.. code:: bash

   rm -rf docs/_build && ./isaaclab.sh --docs


Contributing assets
-------------------

Currently, we host the assets for the extensions on `NVIDIA Nucleus Server <https://docs.omniverse.nvidia.com/nucleus/latest/index.html>`__.
Nucleus is a cloud-based storage service that allows users to store and share large files. It is
integrated with the `NVIDIA Omniverse Platform <https://developer.nvidia.com/omniverse>`__.

Since all assets are hosted on Nucleus, we do not need to include them in the repository. However,
we need to include the links to the assets in the documentation.

The included assets are part of the `Isaac Sim Content <https://docs.isaacsim.omniverse.nvidia.com/latest/assets/index.html>`__.
To use this content, you can use the Asset Browser provided in Isaac Sim.

Please check the `Isaac Sim documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/assets/index.html>`__
for more information on how to download the assets.

.. attention::

  We are currently working on a better way to contribute assets. We will update this section once we
  have a solution. In the meantime, please follow the steps mentioned below.

To host your own assets, the current solution is:

1. Create a separate repository for the assets and add it over there
2. Make sure the assets are licensed for use and distribution
3. Include images of the assets in the README file of the repository
4. Send a pull request with a link to the repository

We will then verify the assets, its licensing, and include the assets into the Nucleus server for hosting.
In case you have any questions, please feel free to reach out to us through e-mail or by opening an issue
in the repository.


Maintaining a changelog and extension.toml
------------------------------------------

Each extension maintains a changelog in the ``CHANGELOG.rst`` file in the ``docs`` directory,
as well as a ``extension.toml`` file in the ``config`` directory.

The ``extension.toml`` file contains the metadata for the extension. It is used to describe the
name, version, description, and other metadata of the extension.

The ``CHANGELOG.rst`` is a file that contains the curated, chronologically ordered list of notable changes
for each version of the extension.

.. note::

   The version number on the ``extension.toml`` file should be updated according to
   `Semantic Versioning <https://semver.org/>`__ and should match the version number in the
   ``CHANGELOG.rst`` file.

The changelog file is written in `reStructuredText <https://docutils.sourceforge.io/rst.html>`__ format.
The goal of this changelog is to help users and contributors see precisely what notable changes have
been made between each release (or version) of the extension. This is a *MUST* for every extension.

For updating the changelog, please follow the following guidelines:

* Each version should have a section with the version number and the release date.
* The version number is updated according to `Semantic Versioning <https://semver.org/>`__. The
  release date is the date on which the version is released.
* Each version is divided into subsections based on the type of changes made.

  * ``Added``: For new features.
  * ``Changed``: For changes in existing functionality.
  * ``Deprecated``: For soon-to-be removed features.
  * ``Removed``: For now removed features.
  * ``Fixed``: For any bug fixes.

* Each change is described in its corresponding sub-section with a bullet point.
* The bullet points are written in the **past tense**.

  * This means that the change is described as if it has already happened.
  * The bullet points should be concise and to the point. They should not be verbose.
  * The bullet point should also include the reason for the change, if applicable.


.. tip::

   When in doubt, please check the style in the existing changelog files and follow the same style.

For example, the following is a sample changelog:

.. code:: rst

    Changelog
    ---------

    0.1.0 (2021-02-01)
    ~~~~~~~~~~~~~~~~~~

    Added
    ^^^^^

    * Added a new feature that helps in a 10x speedup.

    Changed
    ^^^^^^^

    * Changed an existing feature. Earlier, we were using :meth:`torch.bmm` to perform the matrix multiplication.
      However, this was slow for large matrices. We have now switched to using :meth:`torch.einsum` which is
      significantly faster.

    Deprecated
    ^^^^^^^^^^

    * Deprecated an existing feature in favor of a new feature.

    Removed
    ^^^^^^^

    * Removed an existing feature. This was done to simplify the codebase and reduce the complexity.

    Fixed
    ^^^^^

    * Fixed crashing of the :meth:`my_function` when the input was too large.
      We now use :meth:`torch.einsum` that is able to handle larger inputs.


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

To address circular imports, we leverage the `typing.TYPE_CHECKING
<https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING>`_ variable. This special variable is
evaluated only during type-checking, allowing us to import classes or functions in the configuration objects
without triggering circular imports.

It is important to note that this is the sole instance within our codebase where circular imports are used
and are acceptable. In all other scenarios, we adhere to best practices and recommend that you do the same.

Type-hinting
^^^^^^^^^^^^

To make the code more readable, we use `type hints <https://docs.python.org/3/library/typing.html>`__ for
all the functions and classes. This helps in understanding the code and makes it easier to maintain. Following
this practice also helps in catching bugs early with static type checkers like `mypy <https://mypy.readthedocs.io/en/stable/>`__.

**Type-hinting only in the function signature**

To avoid duplication of efforts, we do not specify type hints for the arguments and return values in the docstrings.

For instance, the following are bad examples for various reasons:

.. code:: python

   def my_function(a, b):
      """Adds two numbers.

      This function is a bad example. Reason: No type hints anywhere.

      Args:
         a: The first argument.
         b: The second argument.

      Returns:
         The sum of the two arguments.
      """
      return a + b

.. code:: python

   def my_function(a, b):
      """Adds two numbers.

      This function is a bad example. Reason: Type hints in the docstring and not in the
      function signature.

      Args:
         a (int): The first argument.
         b (int): The second argument.

      Returns:
         int: The sum of the two arguments.
      """
      return a + b

.. code:: python

   def my_function(a: int, b: int) -> int:
      """Adds two numbers.

      This function is a bad example. Reason: Type hints in the docstring and in the function
      signature. Redundancy.

      Args:
         a (int): The first argument.
         b (int): The second argument.

      Returns:
         int: The sum of the two arguments.
      """
      return a + b

The following is how we expect you to write the docstrings and type hints:

.. code:: python

   def my_function(a: int, b: int) -> int:
      """Adds two numbers.

      This function is a good example. Reason: Type hints in the function signature and not in the
      docstring.

      Args:
         a: The first argument.
         b: The second argument.

      Returns:
         The sum of the two arguments.
      """
      return a + b

**No type-hinting for None**

We do not specify the return type of :obj:`None` in the docstrings. This is because
it is not necessary and can be inferred from the function signature.

For instance, the following is a bad example:

.. code:: python

   def my_function(x: int | None) -> None:
      pass

Instead, we recommend the following:

.. code:: python

   def my_function(x: int | None):
      pass

Documenting the code
^^^^^^^^^^^^^^^^^^^^

The code documentation is as important as the code itself. It helps in understanding the code and makes
it easier to maintain. However, more often than not, the documentation is an afterthought or gets rushed
to keep up with the development pace.

**What is considered as a bad documentation?**

* If someone else wants to use the code, they cannot understand the code just by reading the documentation.

  What this means is that the documentation is not complete or is not written in a way that is easy to understand.
  The next time someone wants to use the code, they will have to spend time understanding the code (in the best
  case scenario), or scrap the code and start from scratch (in the worst case scenario).

* Certain design subtleties are not documented and are only apparent from the code.

  Often certain design decisions are made to address specific use cases. These use cases are not
  obvious to someone who wants to use the code. They may change the code in a way that is not intuitive
  and unintentionally break the code.

* The documentation is not updated when the code is updated.

  This means that the documentation is not kept up to date with the code. It is important to update the
  documentation when the code is updated. This helps in keeping the documentation up to date and in sync
  with the code.

**What is considered good documentation?**

We recommend thinking of the code documentation as a living document that helps the reader understand
the *what, why and how* of the code. Often we see documentation that only explains the
what but not the how or why. This is not helpful in the long run.

We suggest always thinking of the documentation from a new user's perspective. They should be able to directly
check the documentation and have a good understanding of the code.

For information on how to write good documentation, please check the notes on
`Dart's effective documentation <https://dart.dev/effective-dart/documentation>`__
and `technical writing <https://en.wikiversity.org/wiki/Technical_writing/Style>`__.
We summarize the key points below:

* Inform (educate the reader) and persuade (convince the reader).
  * Have a clear aim in mind, and make sure everything you write is towards that aim alone.
  * Use examples and analogies before introducing abstract concepts.
* Use the right tone for the audience.
* Compose simple sentences in active voice.
* Avoid unnecessary jargon and repetition. Use plain English.
* Avoid ambiguous phrases such as 'kind of', 'sort of', 'a bit', etc.
* State important information at the beginning of the sentence.
* Say exactly what you mean. Don't avoid writing the uncomfortable truth.


Unit Testing
^^^^^^^^^^^^

We use `pytest <https://docs.pytest.org>`__ for unit testing.
Good tests not only cover the basic functionality of the code but also the edge cases.
They should be able to catch regressions and ensure that the code is working as expected.
Please make sure that you add tests for your changes.

Tools
^^^^^

We use the following tools for maintaining code quality:

* `pre-commit <https://pre-commit.com/>`__: Runs a list of formatters and linters over the codebase.
* `black <https://black.readthedocs.io/en/stable/>`__: The uncompromising code formatter.
* `flake8 <https://flake8.pycqa.org/en/latest/>`__: A wrapper around PyFlakes, pycodestyle and
  McCabe complexity checker.

Please check `here <https://pre-commit.com/#install>`__ for instructions
to set these up. To run over the entire repository, please execute the
following command in the terminal:

.. code:: bash

   ./isaaclab.sh --format  # or "./isaaclab.sh -f"

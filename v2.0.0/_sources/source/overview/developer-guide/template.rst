Building your Own Project
=========================

Traditionally, building new projects that utilize Isaac Lab's features required creating your own
extensions within the Isaac Lab repository. However, this approach can obscure project visibility and
complicate updates from one version of Isaac Lab to another. To circumvent these challenges, we now
provide a pre-configured and customizable `extension template <https://github.com/isaac-sim/IsaacLabExtensionTemplate>`_
for creating projects in an isolated environment.

This template serves three distinct use cases:

* **Project Template**: Provides essential access to Isaac Sim and Isaac Lab's features, making it ideal for projects
  that require a standalone environment.
* **Python Package**: Facilitates integration with Isaac Sim's native or virtual Python environment, allowing for
  the creation of Python packages that can be shared and reused across multiple projects.
* **Omniverse Extension**: Supports direct integration into Omniverse extension workflow.

.. note::

  We recommend using the extension template for new projects, as it provides a more streamlined and
  efficient workflow. Additionally it ensures that your project remains up-to-date with the latest
  features and improvements in Isaac Lab.


Installation
------------

Install Isaac Lab by following the `installation guide <../../setup/installation/index.html>`_. We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

Clone the extension template repository separately from the Isaac Lab installation (i.e. outside the IsaacLab directory):

.. code:: bash

	# Option 1: HTTPS
	git clone https://github.com/isaac-sim/IsaacLabExtensionTemplate.git

	# Option 2: SSH
	git clone git@github.com:isaac-sim/IsaacLabExtensionTemplate.git

Throughout the repository, the name ``ext_template`` only serves as an example and we provide a script to rename all the references to it automatically:

.. code:: bash

	# Enter the repository
	cd IsaacLabExtensionTemplate

	# Rename all occurrences of ext_template (in files/directories) to your_fancy_extension_name
	python scripts/rename_template.py your_fancy_extension_name

Using a python interpreter that has Isaac Lab installed, install the library:

.. code:: bash

	python -m pip install -e source/ext_template


For more details, please follow the instructions in the `extension template repository <https://github.com/isaac-sim/IsaacLabExtensionTemplate>`_.

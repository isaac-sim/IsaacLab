Building your Own Project
=========================

Traditionally, building new projects that utilize Orbit's features required creating your own
extensions within the Orbit repository. However, this approach can obscure project visibility and
complicate updates from one version of Orbit to another. To circumvent these challenges, we now
provide a pre-configured and customizable `extension template <https://github.com/isaac-orbit/orbit.ext_template>`_
for creating projects in an isolated environment.

This template serves three distinct use cases:

* **Project Template**: Provides essential access to Isaac Sim and Orbit's features, making it ideal for projects
  that require a standalone environment.
* **Python Package**: Facilitates integration with Isaac Sim's native or virtual Python environment, allowing for
  the creation of Python packages that can be shared and reused across multiple projects.
* **Omniverse Extension**: Supports direct integration into Omniverse extension workflow.

.. note::

  We recommend using the extension template for new projects, as it provides a more streamlined and
  efficient workflow. Additionally it ensures that your project remains up-to-date with the latest
  features and improvements in Orbit.


To get started, please follow the instructions in the `extension template repository <https://github.com/isaac-orbit/orbit.ext_template>`_.

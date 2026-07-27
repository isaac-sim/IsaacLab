Added
^^^^^

* Added ``Dockerfile.kitless`` and a Compose profile for headless Newton training without Isaac Sim or Kit,
  with OVRTX rendering and all four core reinforcement-learning frameworks.
* Added publishing of the kit-less image to ``nvcr.io/nvidian/isaac-lab-kitless``, using the same branch
  tagging scheme as the Isaac Lab base image, so it can be pulled instead of built locally.

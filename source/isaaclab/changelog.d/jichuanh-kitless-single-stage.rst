Changed
^^^^^^^

* Rebuilt the kit-less Docker image as a single stage. Every wheel the lock resolves for x86_64
  is prebuilt, so the image no longer installs a C toolchain there; ``psutil`` and
  ``pyopengl-accelerate`` have no aarch64 wheels, so the build dependencies they need are
  installed on arm64 only.
* Switched the kit-less image's dependency install to a uv cache mount, matching the Isaac Sim
  images, so a rebuild reuses downloaded wheels instead of re-fetching them.
* Pointed ``uv`` at the shipped environment for runtime commands in the kit-less image, so
  ``uv pip install`` no longer resolves to the externally managed system interpreter.

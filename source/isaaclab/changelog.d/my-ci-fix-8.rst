Fixed
^^^^^

* Restricted the ``pytetwild`` install requirement to x86_64 platforms.
  PyPI ships no aarch64 wheel and the sdist build fails on ARM
  (``-m64`` hardcoded in CMake), which broke the ARM64 docker image
  build.  Tetrahedralization of volume deformables now degrades
  gracefully on ARM64 with the existing "install pytetwild" message
  instead of failing the install outright.
* Made ``./isaaclab.sh -i`` skip the ARM-only swig auto-install when
  ``sudo`` is unavailable instead of crashing with a ``FileNotFoundError``.
  Users on locked-down ARM containers can now run the install and
  pre-provision swig themselves if they need to build nlopt from source.

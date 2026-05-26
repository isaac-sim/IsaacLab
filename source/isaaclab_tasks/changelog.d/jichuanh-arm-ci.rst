Fixed
^^^^^

* Changed the ``ovrtx`` autouse guard in the kitless rendering tests to skip
  rather than fail on aarch64 when the ``ov[ovrtx]`` optional dependency is
  unavailable. The ``ovrtx`` wheel is published only for x86_64, so on aarch64
  this gate was turning unreachable parametrize cases into hard failures; x86
  environments without ``ov[ovrtx]`` still see the original "install with
  ``./isaaclab.sh -i 'ov[ovrtx]'``" failure with install guidance.

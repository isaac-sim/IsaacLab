Fixed
^^^^^

* Changed the kitless rendering tests' ``ov[ovrtx]`` and ``ov[ovphysx]`` autouse
  guards to skip rather than fail on aarch64 when the optional dependency is
  unavailable. Both wheels are published only for x86_64, so on aarch64 these
  gates were turning unreachable parametrize cases into hard failures; x86
  environments without the dependency still see the original
  "install with ``./isaaclab.sh -i 'ov[…]'``" failure with install guidance.

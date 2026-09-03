Fixed
^^^^^

* Fixed pretrained checkpoint resolution for coupled tasks such as ``Isaac-Lift-Cable-Franka``,
  ``Isaac-Lift-Cloth-Franka``, and ``Isaac-Lift-Soft-Franka``, which raised
  ``Unsupported Newton solver for pretrained checkpoints: CouplerProxyCfg``. A Newton coupled
  solver is now named by its entry solvers in order followed by its coupling scheme, so a proxy
  coupler over MJWarp and VBD entries resolves to the ``newtonmjwarpvbdproxy`` physics token.
  Checkpoint names for uncoupled solvers are unchanged.

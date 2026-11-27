* Fixed ``scripts/neural/train.py`` and the :class:`~isaaclab_neural.physics.NeRDSolverCfg` and
  :class:`~isaaclab_neural.physics.MJWarpNeRDSolverCfg` physics presets failing on Windows with
  ``TritonMissing``. ``torch.compile`` needs Triton to generate CUDA code, and PyTorch installs it on Linux
  only, so a model now trains and runs without ``torch.compile`` by default where Triton is missing.

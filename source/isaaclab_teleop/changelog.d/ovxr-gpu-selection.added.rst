* Added :attr:`~isaaclab_teleop.OvxrRendererCfg.active_cuda_gpus` and the ``--ovrtx_active_cuda_gpus`` OVXR
  option of live teleoperation and recording to render the session on GPUs other than the physics GPU.
  Rigid transforms and deformable points from the simulation GPU are published through completed copies
  on the first rendering GPU. Malformed values are rejected when the configuration is created. OVRTX
  selects rendering GPUs once per process, so the OVXR renderer rejects a later session that requests a
  different selection, including after teardown or failed startup. Ordinary OVRTX cameras still render on
  the simulation device: a session that renders on other GPUs rejects one registered in the same
  simulation, but not one in another simulation of the same process.
* Added ``python -m isaaclab_teleop.tools.check_cuda_peer``, a preflight that checks the geometry publication
  copy between two GPUs before splitting physics and rendering across them.

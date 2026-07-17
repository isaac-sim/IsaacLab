Fixed
^^^^^

* Fixed :meth:`~isaaclab.envs.DirectRLEnv.reset` to store the observation buffer
  like :meth:`~isaaclab.envs.DirectRLEnv.step` already does, and exposed the
  latest observations on the multi-agent-to-single-agent adapter through the
  same public buffer.
* Fixed kit-less installation to select one OpenUSD provider per architecture,
  preventing mixed ``pxr`` ABIs during Newton training on x86.

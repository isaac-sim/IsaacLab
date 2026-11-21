Added OptiX cameras to :class:`~isaaclab_experimental.envs.MDP2Env`: cameras with
:class:`~isaaclab_newton.renderers.NewtonOptixRendererCfg` render between the CUDA graphs of a step, since OptiX
refits its acceleration structures on the host.

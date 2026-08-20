Fixed
^^^^^
* Reduced the default number of parallel environments for GearAssembly tasks to 1024 so recurrent PPO training fits on development GPUs. Use ``--num_envs`` to select a different batch size.

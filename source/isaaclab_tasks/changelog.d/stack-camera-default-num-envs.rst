Fixed
^^^^^

* Fixed camera-enabled stack environments exhausting GPU memory with their inherited default environment count by
  defaulting those tasks to one environment. Use ``--num_envs`` to configure a larger batch.

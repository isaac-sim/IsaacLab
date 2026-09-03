Fixed
^^^^^

* Fixed stack environments exhausting GPU memory with their inherited default environment count by defaulting the
  shared stack environment configurations to one environment. Use ``--num_envs`` to configure a larger batch.

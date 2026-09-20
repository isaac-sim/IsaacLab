Changed
^^^^^^^

* **Breaking:** Declared normalized ``[-1, 1]`` raw policy action bounds for the maintained Stable-Baselines3
  environments. Policies sending values outside this range are now clipped before action scaling.

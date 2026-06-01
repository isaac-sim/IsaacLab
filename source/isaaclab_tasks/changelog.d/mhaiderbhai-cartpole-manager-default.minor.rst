Changed
^^^^^^^

* **Breaking:** Dropped the ``-Manager`` suffix from the manager-based cartpole task IDs so the
  default (manager-based) workflow carries no workflow suffix; the direct-workflow tasks keep
  their explicit ``-Direct`` suffix. Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Cartpole-Manager`` → ``Isaac-Cartpole``.
  * ``Isaac-Cartpole-Camera-Manager`` → ``Isaac-Cartpole-Camera``.

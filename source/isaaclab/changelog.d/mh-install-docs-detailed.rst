Removed
^^^^^^^

* **Breaking:** Removed the aggregate ``all`` extra. Request the extras you need by name
  instead, for example ``uv pip install "isaaclab[isaacsim,sb3,skrl,rsl-rl]"`` in place of
  ``uv pip install "isaaclab[isaacsim,all]"``, or ``uv run --extra sb3 --extra skrl`` in place
  of ``uv run --extra all``. See :ref:`installation-optional-extras` for the full extras table.

Added
^^^^^

* Added cached rigid-transform requests to the scene data provider. Render integrations can use
  ``request_transforms(SceneDataFormat.Transform)`` to share native pointers, or request a different
  format and ordering with one conversion per dirty publication. The existing ``get_transforms``
  API remained available for caller-owned outputs.

Changed
^^^^^^^

* **Breaking:** Renderer integrations used ``SceneDataBackend.transform_publication`` for native
  pointers and dirty state. Custom physics backends must expose a ``SceneDataPublication`` and mark
  it dirty after state writes, simulation steps, and native buffer swaps.

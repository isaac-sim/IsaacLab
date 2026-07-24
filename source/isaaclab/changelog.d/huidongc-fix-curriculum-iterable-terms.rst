Fixed
^^^^^

* Fixed :meth:`~isaaclab.managers.CurriculumManager.get_active_iterable_terms`
  crashing with ``TypeError`` when a curriculum term state is a ``dict``, which
  broke Newton visualizer live plots during training.

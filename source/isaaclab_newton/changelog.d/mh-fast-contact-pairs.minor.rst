Added
^^^^^

* Added :class:`~isaaclab_newton.physics.NewtonModelBuilder`, a Newton
  ``ModelBuilder`` whose finalize-time contact-pair search is vectorized with
  numpy while producing identical pairs in identical order.
  :meth:`~isaaclab_newton.physics.NewtonManager.create_builder` and the Newton
  visualization builder now use it, reducing model finalize time by roughly a
  third for large environment counts.

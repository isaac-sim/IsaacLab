Fixed
^^^^^

* Fixed :func:`~isaaclab.cloner.usd_replicate` authoring missing intermediate
  ancestors of nested clone destinations (for example the ``Groceries`` scope in
  ``/World/envs/env_{}/Groceries/Object``) as ``over`` prim specs. The copied
  prims composed as undefined, so their references never expanded and renderers
  skipped them in every environment except the source one.

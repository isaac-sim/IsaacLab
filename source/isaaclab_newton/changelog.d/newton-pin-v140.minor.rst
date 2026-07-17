Changed
^^^^^^^

* Changed the pinned Newton version to the v1.4.0 release (from a pre-1.4
  development commit), adopting the MuJoCo 3.10 stack and the new margin/gap
  semantics. Assets that author PhysX ``contactOffset`` / ``restOffset``
  attributes keep their behavior through Newton's schema translation
  (``margin == restOffset``, ``gap == contactOffset - restOffset``).

Fixed
^^^^^

* Fixed custom-frequency USD traversal honoring ``ignore_paths`` via the
  upstream Newton fix included in v1.4.0.

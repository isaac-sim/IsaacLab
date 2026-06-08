Changed
^^^^^^^

* Switched the Newton install spec to ``newton[sim]`` in the ``newton``,
  ``rerun``, and ``viser`` extras so the MuJoCo solver dependencies are
  pulled in transitively. Required because pip resolves a git-URL
  requirement once for the URL; a bare ``newton @ git+...`` here would
  shadow the ``[sim]`` extra requested elsewhere.

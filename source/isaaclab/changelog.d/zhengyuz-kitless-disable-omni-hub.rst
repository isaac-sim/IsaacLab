Fixed
^^^^^

* Fixed kitless runs auto-launching a globally installed Omniverse Hub to read USD assets.
  ``OMNICLIENT_HUB_MODE`` now defaults to ``disabled`` when Kit is not started, and setting the
  variable explicitly still selects the shared or exclusive mode.

Changed
^^^^^^^

* Changed remote asset retrieval to download HTTP(S) assets directly when ``omni.client`` is not
  installed, so kit-less installs without ``omniverseclient`` wheels, such as macOS, can load the
  default asset root.

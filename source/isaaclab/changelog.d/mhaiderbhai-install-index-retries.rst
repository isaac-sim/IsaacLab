Fixed
^^^^^

* Fixed ``isaaclab -i`` aborting on transient package index errors by raising uv's HTTP retry
  budget from its default of 3 retries to 6. Set ``UV_HTTP_RETRIES`` to override.

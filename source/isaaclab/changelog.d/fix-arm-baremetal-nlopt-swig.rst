Fixed
^^^^^

* Fixed native (non-Docker) ARM installation failing when ``swig`` is missing.
  The ``nlopt==2.6.2`` source build now installs ``swig`` via apt only for the
  duration of the build and purges it afterwards, so the GPL-licensed ``swig``
  is never left behind — in particular it is never shipped in the Docker image.

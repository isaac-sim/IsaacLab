Fixed
^^^^^

* Fixed Linux ARM Docker installation by keeping ``swig`` available while
  building ``nlopt==2.6.2`` and installing Isaac Lab dependencies, then
  removing ``swig`` before the image layer completes.

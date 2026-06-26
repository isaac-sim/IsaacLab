Fixed
^^^^^

* Fixed Linux ARM installation by pre-installing ``nlopt==2.6.2`` before
  Isaac Lab dependencies are resolved, and by keeping ``swig`` available in
  Docker only until dependency installation completes.

Fixed
^^^^^

* Fixed ``./isaaclab.sh --install`` unnecessarily building ``nlopt==2.6.2`` on
  ARM Linux (e.g. DGX Spark), where it fails with CMake 4.x. The ARM-only
  ``nlopt`` pre-install and its temporary ``swig`` install were removed because
  ``nlopt`` is only pulled in by Linux x86_64 dependencies.

* Fixed unavoidable ARM source builds such as ``egl-probe==1.0.2`` failing when
  CMake 4.x rejects policy compatibility older than 3.5. The installer now
  temporarily sets ``CMAKE_POLICY_VERSION_MINIMUM=3.5`` during ARM dependency
  builds.

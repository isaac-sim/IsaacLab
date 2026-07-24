Fixed
^^^^^

* Fixed ``./isaaclab.sh --install`` failing on ARM Linux (e.g. DGX Spark) when building
  ``nlopt==2.6.2`` from source with CMake 4.x, which no longer supports the
  ``cmake_minimum_required`` value declared by that release. The ARM-only ``nlopt``
  pre-install (and its temporary ``swig`` install) was removed because ``nlopt`` is only
  pulled in by ``isaacteleop`` and ``dex-retargeting``, which are Linux x86_64 only.

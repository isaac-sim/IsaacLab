Fixed
^^^^^

* Fixed environment arguments being overwritten after reopening an HDF5 dataset and leaking between datasets when
  reusing a file handler.

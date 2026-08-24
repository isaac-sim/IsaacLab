Fixed
^^^^^

* Fixed the HDF5 merge and MP4 augmentation tools preserving the dataset ``format_version`` so current XYZW datasets remain correctly labeled through the documented augmentation-and-merge workflow. Mixed legacy/current inputs are rejected because one output file cannot safely represent multiple root-pose quaternion formats.

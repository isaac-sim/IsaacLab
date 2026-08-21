Fixed
^^^^^

* Fixed the HDF5 merge tool dropping the dataset ``format_version`` and causing current XYZW datasets to be treated as legacy WXYZ data after merging. Mixed legacy/current inputs are now rejected because one output file cannot safely represent multiple root-pose quaternion formats.

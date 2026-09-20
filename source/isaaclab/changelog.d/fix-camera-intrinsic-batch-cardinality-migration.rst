Fixed
-----

Camera intrinsic updates now reject mismatched batches before writing USD data. Callers must provide one matrix per selected camera or narrow ``env_ids`` to match the supplied matrices.

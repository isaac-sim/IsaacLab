Fixed
^^^^^

* Fixed ``convert_dict_to_backend`` resetting nested dictionaries to the default NumPy backend instead
  of preserving the backend and source array types requested by the caller.

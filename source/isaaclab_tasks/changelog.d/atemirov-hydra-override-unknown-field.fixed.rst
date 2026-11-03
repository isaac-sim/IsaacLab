* Fixed command-line overrides of configuration class fields silently adding a misspelled field and leaving the
  intended field unchanged. Such an override now warns with the override and the missing field. Keys of
  dictionaries, such as a dictionary-based agent configuration, are not checked.

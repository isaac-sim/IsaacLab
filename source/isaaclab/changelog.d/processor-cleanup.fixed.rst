* Fixed manager-based environment teardown to finish releasing resources when an observation
  processor raised during cleanup, while preserving the error and allowing repeated close calls.

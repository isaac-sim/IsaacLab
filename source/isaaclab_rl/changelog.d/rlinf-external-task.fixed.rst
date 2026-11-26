* Fixed RLinf training and playback to pass the CLI task override to worker registration.
* Preserved split-specific environment overrides from RLinf recipes when merging command-line overrides, replacing conflicting recipe keys including backend preset selectors.
* Rejected incompatible policy image settings before RLinf worker startup so extension loading cannot hide configuration errors.

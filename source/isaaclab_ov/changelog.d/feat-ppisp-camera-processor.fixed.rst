* Fixed ``rgb_hdr`` extraction after OVRTX falls back from the requested CUDA device. ``HdrColor`` tiles are
  now copied to the output device for every ``rgb_hdr`` request, not only when PPISP was enabled.

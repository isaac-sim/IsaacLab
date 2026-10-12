* Defaulted the launcher's parsed ``visualizer`` selection to an empty list instead of ``None``.
  Omission of ``--viz`` continued to select no display; code inspecting the parsed arguments could iterate
  the selection directly.

* Added ``SteelSolverCfg`` and ``SteelSolver`` to select Newton's experimental Steel rigid-body solver through the existing Newton physics composition root, with point friction as the default.
* Exposed Steel's ``use_parallel_streams`` option so task presets can select its specialized CUDA paths.
* Added Steel options for parallel tree traversal and fixed-root contact filtering.

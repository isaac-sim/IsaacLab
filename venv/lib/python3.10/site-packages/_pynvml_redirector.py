# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import importlib.abc
import sys
import warnings

PYNVML_MSG = (
    "The pynvml package is deprecated. Please install nvidia-ml-py instead. "
    "If you did not install pynvml directly, please report this to the maintainers of "
    "the package that installed pynvml for you."
)


PYNVML_UTILS_MSG = (
    "The pynvml_utils module is deprecated. The source will remain accessible on "
    "Github at https://github.com/gpuopenanalytics/pynvml/."
)


class PynvmlFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.has_warned_pynvml = False
        self.has_warned_pynvml_utils = False

    def find_spec(self, name, _, __=None):
        if name.startswith("pynvml") and not self.has_warned_pynvml:
            warnings.warn(PYNVML_MSG, FutureWarning, stacklevel=2)
            self.has_warned_pynvml = True

        if name.startswith("pynvml_utils") and not self.has_warned_pynvml_utils:
            warnings.warn(PYNVML_UTILS_MSG, FutureWarning, stacklevel=2)
            self.has_warned_pynvml_utils = True
        # Defer actually finding the module to the other finders


finder = PynvmlFinder()
sys.meta_path.insert(0, finder)

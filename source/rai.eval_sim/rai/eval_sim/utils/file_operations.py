# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import ast
import importlib
import inspect
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any

from .log import log_error

# Get path to assets
EVAL_SIM_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EVAL_SIM_DIR.parents[1]
ASSETS_DIR = REPO_DIR / "assets"
CONFIG_DIR = REPO_DIR / "config"
OUTPUT_DIR = REPO_DIR / "output"
TASKS_DIR = EVAL_SIM_DIR / "tasks"
GAMEPAD_CFG_PATH = EVAL_SIM_DIR / "teleop" / "gamepad.yaml"
USER_EVAL_SIM_CFG_PATH = REPO_DIR / "user_eval_sim_cfg.yaml"


def import_class_dynamically(file_path: str, class_name: str) -> type:
    """Import a class from a file dynamically.

    Args:
        file_path: The path to the file containing the class.
        class_name: The name of the class to import.

    Returns:
        cls: The imported class.

    """
    # fnsure the file's directory is in sys.path
    file_dir = os.path.dirname(file_path)
    if file_dir not in sys.path:
        sys.path.append(file_dir)

    # convert the file path to a module path
    module_path = os.path.splitext(os.path.basename(file_path))[0]

    # import the module
    module = importlib.import_module(module_path)

    # get the class from the module
    cls = getattr(module, class_name, None)

    return cls


def extract_classes(file_path: str) -> list[str]:
    """Extracts all classes from a Python file.

    Args:
        file_path: The path to the file to extract classes from.

    Returns:
        classes: A list of class names extracted from the file

    """
    with open(file_path) as file:
        tree = ast.parse(file.read())

    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

    return classes


def find_env_cfgs_in_file(file_path: str) -> str:
    """Finds ManagerBasedEnvCfg and ManagerBasedRLEnvCfg classes in a file.

    Args:
        file_path: The path to the file to search for env cfgs.

    Returns:
        class_name: The name of the class found.

    Raises:
        ValueError: If no class name ending in "EnvCfg" is found.

    """
    classes = extract_classes(file_path)
    for class_name in classes:
        if class_name.endswith("EnvCfg"):
            return class_name
    raise ValueError("No classname found ending with EnvCfg.")


def find_ros_manager_cfgs_in_file(file_path: str) -> str:
    classes = extract_classes(file_path)
    for class_name in classes:
        if class_name.endswith("RosManagerCfg"):
            return class_name
    raise ValueError("No classname found ending with RosManagerCfg.")


def import_all_modules(package_name) -> dict[str, Any]:
    """Dynamically imports all modules in the specified package name.

    Args:
        package_name: The package name to search for modules.

    Returns:
        imported_modules: A dictionary mapping module names to the module.
    """
    imported_modules = {}
    # Attempt to import the package itself first
    package = importlib.import_module(package_name)

    # Check all submodules and subpackages in the package
    for _, name, _ in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
        # Dynamically import the module
        try:
            module = importlib.import_module(name)
            imported_modules[name] = module
        except ImportError as e:
            log_error(f"Failed to import {package}: {e.msg}")
        except Exception as e:
            log_error(f"Error processing {package}: {e}")

    return imported_modules


def find_subclasses_of_base(
    packages: list[str], base_class: type, ignore_classes: list[Any] = list()
) -> dict[str, list[str]]:
    """Finds and returns subclasses of a specified base class within given packages.

    This function searches through specified packages, importing all modules within them,
    and identifies all subclasses of a given base class. It can optionally ignore specific classes.

    Args:
        packages: A list of package names (as strings) to search for subclasses.
        base_class: The base class to search for subclasses of.
        ignore_classes: A list of classes to ignore during the search. Defaults to an empty list.

    Returns:
        subclasses: A dictionary where keys are module and class names (as strings)
            and values are the corresponding class objects that are subclasses of the base class.

    """

    subclasses = {}
    for package in packages:
        try:
            for module_name, module in import_all_modules(package).items():
                try:
                    # Iterate over all members of the module
                    for name, obj in inspect.getmembers(module):
                        # Check if the member is a class and it is not the base_class itself
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, base_class)
                            and obj not in [base_class] + ignore_classes
                        ):
                            subclasses[obj.__module__ + "." + name] = obj
                except ImportError:
                    log_error(f"Failed to import {package}")
                except Exception as e:
                    log_error(f"Error processing {package}: {e}")
        except ModuleNotFoundError:
            log_error(f"Failed to import {package}. Double check that it exists and is installed.")
    return subclasses

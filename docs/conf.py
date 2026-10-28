# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys

import tomllib

sys.path.insert(0, os.path.abspath("_extensions"))
sys.path.insert(0, os.path.abspath("../source/isaaclab"))
sys.path.insert(0, os.path.abspath("../source/isaaclab/isaaclab"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_assets"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_assets/isaaclab_assets"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_tasks"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_tasks/isaaclab_tasks"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_tasks_experimental"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_tasks_experimental/isaaclab_tasks_experimental"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_physx"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_physx/isaaclab_physx"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_newton"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_newton/isaaclab_newton"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_experimental"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_experimental/isaaclab_experimental"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_rl"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_rl/isaaclab_rl"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_mimic"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_mimic/isaaclab_mimic"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_contrib"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_contrib/isaaclab_contrib"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_teleop"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_teleop/isaaclab_teleop"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_ov"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_ov/isaaclab_ov"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_visualizers"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_visualizers/isaaclab_visualizers"))

# -- Project information -----------------------------------------------------

project = "Isaac Lab"
copyright = "2022-2025, The Isaac Lab Project Developers."
author = "The Isaac Lab Project Developers."

# Read version from the package
with open(os.path.join(os.path.dirname(__file__), "..", "VERSION")) as f:
    full_version = f.read().strip()
    version = ".".join(full_version.split(".")[:3])

# Default development branch referenced by installation documentation.
isaaclab_latest_branch = os.getenv("ISAACLAB_LATEST_BRANCH", "main")
isaaclab_wheel_version = "3.0.0"
isaaclab_wheel_source_tag = "v3.0.0"


def _read_pinned_versions() -> dict:
    """Read the ``[tool.isaaclab.versions]`` table from the root pyproject.

    This table is the single source of truth for externally-pinned versions
    (Isaac Sim, the torch stack, the OV renderer/physics wheels).
    """
    pyproject = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject, "rb") as f:
        return tomllib.load(f)["tool"]["isaaclab"]["versions"]


# Pinned external versions referenced by the installation docs. Shared with the
# ``isaaclab_docs`` extension via config values of the same name.
_pinned_versions = _read_pinned_versions()
isaacsim_version = _pinned_versions["isaacsim"]
torch_version = _pinned_versions["torch"]
torchvision_version = _pinned_versions["torchvision"]
ovrtx_version = _pinned_versions["ovrtx"]
ovrtx_spec = f"=={ovrtx_version}" if ovrtx_version[0].isdigit() else ovrtx_version
ovphysx_version = _pinned_versions["ovphysx"]

# Short version strings used in external documentation URLs and badges.
torch_docs_version = ".".join(torch_version.split(".")[:2])  # e.g. "2.11"
isaacsim_docs_version = ".".join(isaacsim_version.split(".")[:3])  # e.g. "6.2.0"
# The rendered OVRTX docs are unversioned, so link its docs sources at the pinned release tag, e.g. "v0.5.1".
ovrtx_docs_ref = f"v{'.'.join(ovrtx_version.split('.')[:3])}" if ovrtx_version[0].isdigit() else "main"

# Copy buttons on highlighted code blocks (including nested directive output).
copybutton_selector = "div.highlight pre"

rst_prolog = f"""
.. |isaaclab_latest_branch| replace:: {isaaclab_latest_branch}
.. |isaaclab_wheel_version| replace:: {isaaclab_wheel_version}
.. |isaacsim_version| replace:: {isaacsim_version}
.. |torch_version| replace:: {torch_version}
.. |torchvision_version| replace:: {torchvision_version}
.. |ovrtx_spec| replace:: {ovrtx_spec}
.. |ovrtx_docs_ref| replace:: {ovrtx_docs_ref}
.. |ovphysx_version| replace:: {ovphysx_version}
"""

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autodocsumm",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.linkcode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.icon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_paramlinks",
    "sphinx_tabs.tabs",  # backwards compatibility for building docs on v1.0.0
    "sphinx_multiversion",
    "isaaclab_docs",
]

# mathjax hacks
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

# panels hacks
panels_add_bootstrap_css = False
panels_add_fontawesome_css = True

# supported file extensions for source files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# make sure we don't have any unknown references
# TODO: Enable this by default once we have fixed all the warnings
# nitpicky = True

nitpick_ignore = [
    ("py:obj", "slice(None)"),
]

nitpick_ignore_regex = [
    (r"py:.*", r"pxr.*"),  # we don't have intersphinx mapping for pxr
    (r"py:.*", r"trimesh.*"),  # we don't have intersphinx mapping for trimesh
]

# emoji style
sphinxemoji_style = "twemoji"  # options: "twemoji" or "unicode"
# put type hints inside the signature instead of the description (easier to maintain)
autodoc_typehints = "signature"
# autodoc_typehints_format = "fully-qualified"
# document class *and* __init__ methods
autoclass_content = "class"  #
# separate class docstring from __init__ docstring
autodoc_class_signature = "separated"
# sort members by source order
autodoc_member_order = "bysource"
# inherit docstrings from base classes
autodoc_inherit_docstrings = True
# BibTeX configuration
bibtex_bibfiles = ["source/_static/refs.bib"]
# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = False
# default autodoc settings
autodoc_default_options = {
    "autosummary": True,
}

# generate links to the documentation of objects in external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "trimesh": ("https://trimesh.org/", None),
    # pinned to the release version because /docs/stable/objects.inv currently 404s
    "torch": (f"https://docs.pytorch.org/docs/{torch_docs_version}/", None),
    # Versioned documentation can lag a newly published Isaac Sim package release.
    "isaacsim": (
        "https://docs.isaacsim.omniverse.nvidia.com/latest/py/",
        "https://docs.isaacsim.omniverse.nvidia.com/latest/py/objects.inv",
    ),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    # NOTE: pinned to /stable/ because /objects.inv at the root currently 404s
    "warp": ("https://nvidia.github.io/warp/stable/", None),
    "omniverse": ("https://docs.omniverse.nvidia.com/dev-guide/latest", None),
}

# Usage: :ovrtx-docs:`Minimal mode <sensors/cameras/render_modes/minimal.rst>`
extlinks = {
    "ovrtx-docs": (f"https://github.com/NVIDIA-Omniverse/ovrtx/blob/{ovrtx_docs_ref}/docs/%s", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "_redirect",
    "_templates",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "licenses/*",
    "plans",
    # Include-only fragments (pulled in via ``.. include::``; not standalone pages).
]

# Mock out modules that are not available on RTD
# Warp stays real: generated kernels resolve concrete element layouts during import,
# without compiling or launching a GPU kernel. Warp is a normal project dependency.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "matplotlib",
    "scipy",
    "carb",
    "pxr",
    "isaacsim",
    "omni",
    "omni.kit",
    "omni.log",
    "omni.usd",
    "omni.client",
    "omni.physx",
    "omni.physics",
    "ovphysx",
    "usdrt",
    "pxr.PhysxSchema",
    "pxr.PhysicsSchemaTools",
    "omni.replicator",
    "isaacsim.core.cloner",
    "isaacsim.core.version",
    "isaacsim.core.experimental.prims",
    "isaacsim.core.experimental.utils",
    "isaacsim.core.rendering_manager",
    "isaacsim.robot_motion.motion_generation",
    "isaacsim.gui.components",
    "isaacsim.asset.importer.urdf",
    "isaacsim.asset.importer.mjcf",
    "omni.syntheticdata",
    "omni.timeline",
    "omni.ui",
    "gym",
    "gymnasium",
    "skrl",
    "stable_baselines3",
    "rsl_rl",
    "rl_games",
    "ray",
    "h5py",
    "hid",
    "prettytable",
    "psutil",
    "tqdm",
    "tensordict",
    "torchrl",
    "trimesh",
    "toml",
    "pink",
    "pinocchio",
    "qpsolvers",
    "flatdict",
    "leapp",
    "ovrtx",
    "filelock",
    "IPython",
    "cv2",
    "imageio",
    "ipywidgets",
    "mpl_toolkits",
    "isaacteleop",
    "scipy",
    "hydra",
    "hydra.core",
    "hydra.core.config_store",
    "omegaconf",
    "newton",
]

# List of zero or more Sphinx-specific warning categories to be squelched (i.e.,
# suppressed, ignored).
suppress_warnings = [
    # Generally speaking, we do want Sphinx to inform
    # us about cross-referencing failures. Remove this entirely after Sphinx
    # resolves this open issue:
    #   https://github.com/sphinx-doc/sphinx/issues/4961
    # Squelch mostly ignorable warnings resembling:
    #     WARNING: more than one target found for cross-reference 'TypeHint':
    #     beartype.door._doorcls.TypeHint, beartype.door.TypeHint
    #
    # Sphinx currently emits *MANY* of these warnings against our
    # documentation. All of these warnings appear to be ignorable. Although we
    # could explicitly squelch *SOME* of these warnings by canonicalizing
    # relative to absolute references in docstrings, Sphinx emits still others
    # of these warnings when parsing PEP-compliant type hints via static
    # analysis. Since those hints are actual hints that *CANNOT* by definition
    # by canonicalized, our only recourse is to squelch warnings altogether.
    "ref.python",
]

# -- Internationalization ----------------------------------------------------

# specifying the natural language populates some key tags
language = "en"

# -- Options for HTML output -------------------------------------------------

import sphinx_book_theme

html_title = "Isaac Lab Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_favicon = "source/_static/favicon.ico"
html_show_copyright = True
html_show_sphinx = False
html_last_updated_fmt = ""  # to reveal the build date in the pages meta

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["source/_static"]
html_css_files = [
    "css/custom.css",
    "css/environment-browser.css",
    "css/demo-browser.css",
    "css/guide-browser.css",
]
html_js_files = [
    "css/environment-browser.js",
    "css/demo-browser.js",
    "css/guide-browser.js",
]

html_theme_options = {
    "announcement": (
        '<a href="https://docs.nvidia.com/isaaclab">Explore our AI-friendly Fern documentation</a>'
        " &mdash; Chat with AI about Isaac Lab and access machine-readable docs."
    ),
    "path_to_docs": "docs/",
    "navbar_persistent": [],
    "collapse_navigation": True,
    "repository_url": "https://github.com/isaac-sim/IsaacLab",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "use_sidenotes": True,
    "logo": {
        "text": "Isaac Lab Documentation",
        "image_light": "source/_static/NVIDIA-logo-white.png",
        "image_dark": "source/_static/NVIDIA-logo-black.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/isaac-sim/IsaacLab",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Isaac Sim",
            "url": "https://developer.nvidia.com/isaac-sim",
            "icon": f"https://img.shields.io/badge/IsaacSim-{isaacsim_docs_version}-silver.svg",
            "type": "url",
        },
        {
            "name": "Stars",
            "url": "https://img.shields.io/github/stars/isaac-sim/IsaacLab?color=fedcba",
            "icon": "https://img.shields.io/github/stars/isaac-sim/IsaacLab?color=fedcba",
            "type": "url",
        },
    ],
    "icon_links_label": "Quick Links",
}

templates_path = [
    "_templates",
]

# Whitelist pattern for remotes
smv_remote_whitelist = r"^.*$"
# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = os.getenv("SMV_BRANCH_WHITELIST", r"^(main|develop)$")
# Whitelist pattern for tags (set to None to ignore all tags).
# Matches vMAJOR.MINOR.PATCH with an optional pre-release suffix like -beta or -rc1,
# so tags like v3.0.0-beta show up in the version selector.
smv_tag_whitelist = os.getenv("SMV_TAG_WHITELIST", r"^v[1-9]\d*\.\d+\.\d+(-[A-Za-z0-9.]+)?$")
html_sidebars = {
    "**": ["navbar-logo.html", "versioning.html", "icon-links.html", "search-field.html", "sbt-sidebar-nav.html"]
}


# Keep published links working after guide consolidation.
isaaclab_doc_redirects = {
    "source/concepts/schema_fragments": "source/concepts/schema_cfgs",
    "source/migration/migrating_to_isaaclab_3-0": "source/isaaclab/migrating",
    "source/setup/ecosystem": "source/isaaclab/ecosystem",
    "source/setup/environments": "source/isaaclab/environments",
    "source/setup/quickstart": "source/isaaclab/quickstart",
    "source/setup/tutorial": "source/isaaclab/tutorial",
    "source/setup/demos": "source/isaaclab/demos",
    "source/setup/interactive_demos": "source/isaaclab/interactive_demos",
    "source/setup/installation/index": "source/isaaclab/installation",
    "source/concepts/backend_architecture": "source/concepts/backends_and_presets",
    "source/concepts/joint_and_body_ordering": "source/how-to/transfer_policies_between_physx_and_newton",
    "source/concepts/native-physics-api/newton": "source/concepts/physics-backends/newton",
    "source/concepts/native-physics-api/ovphysx": "source/concepts/physics-backends/ovphysx",
    "source/concepts/native-physics-api/physx": "source/concepts/physics-backends/physx",
    "source/concepts/ovphysx": "source/concepts/physics-backends/ovphysx",
    "source/concepts/warp_environment_migration": "source/concepts/warp_environments",
    "source/concepts/solver-tuning/index": "source/how-to/index",
    "source/concepts/solver-tuning/tune_kamino": "source/how-to/solver_tuning_kamino",
    "source/concepts/solver-tuning/tune_mjwarp": "source/how-to/solver_tuning_mjwarp",
    "source/concepts/solver-tuning/tune_mpm": "source/how-to/solver_tuning_mpm",
    "source/concepts/solver-tuning/tune_physx": "source/how-to/solver_tuning_physx",
    "source/concepts/solver-tuning/tune_vbd": "source/how-to/solver_tuning_vbd",
    "source/features/docker_cloud": "source/isaaclab/installation",
    "source/workflows/index": "source/isaaclab/installation",
    "source/workflows/docker/index": "source/isaaclab/installation",
    "source/workflows/docker/images": "source/isaaclab/installation",
    "source/workflows/docker/cluster": "source/features/hpc_clusters",
    "source/workflows/docker/cloud": "source/features/cloud_deployment",
    "source/features/reproducibility": "source/how-to/reproducibility",
    "source/concepts/native-physics-api/index": "source/concepts/physics_backends",
    "source/concepts/native-physics-api/scene_data_providers": "source/concepts/scene_data_providers",
    "source/developer-tools/scene_data_providers": "source/concepts/scene_data_providers",
    "source/features/draw_markers": "source/how-to/create_visualization_markers",
    "source/features/record_video": "source/how-to/record_video",
    "source/features/visualizer_tiled_camera": "source/how-to/visualizer_streaming_camera_view",
    "source/how-to/robots": "source/how-to/write_articulation_cfg",
    "source/tutorials/00_sim/create_empty": "source/how-to/create_empty",
    "source/tutorials/00_sim/launch_app": "source/how-to/launch_app",
    "source/tutorials/00_sim/spawn_prims": "source/how-to/spawn_prims",
    "source/tutorials/01_assets/add_new_robot": "source/how-to/write_articulation_cfg",
    "source/tutorials/01_assets/run_articulation": "source/how-to/run_articulation",
    "source/tutorials/01_assets/run_deformable_object": "source/how-to/run_deformable_object",
    "source/tutorials/01_assets/run_rigid_object": "source/how-to/run_rigid_object",
    "source/tutorials/01_assets/run_surface_gripper": "source/how-to/run_surface_gripper",
    "source/tutorials/02_scene/create_scene": "source/how-to/create_scene",
    "source/tutorials/03_envs/configuring_rl_training": "source/how-to/configuring_rl_training",
    "source/tutorials/03_envs/create_direct_rl_env": "source/how-to/create_direct_rl_env",
    "source/tutorials/03_envs/create_manager_base_env": "source/how-to/create_manager_base_env",
    "source/tutorials/03_envs/create_manager_rl_env": "source/how-to/create_manager_rl_env",
    "source/tutorials/03_envs/modify_direct_rl_env": "source/how-to/modify_direct_rl_env",
    "source/tutorials/03_envs/policy_inference_in_usd": "source/how-to/policy_inference_in_usd",
    "source/tutorials/03_envs/register_rl_env_gym": "source/how-to/register_rl_env_gym",
    "source/tutorials/03_envs/run_rl_training": "source/how-to/run_rl_training",
    "source/tutorials/04_sensors/add_sensors_on_robot": "source/how-to/add_sensors_on_robot",
    "source/tutorials/05_controllers/run_diff_ik": "source/how-to/run_diff_ik",
    "source/tutorials/05_controllers/run_osc": "source/how-to/run_osc",
    "source/tutorials/index": "source/how-to/index",
    # Routes published by the 2.x main branch before the 3.x documentation reorganization.
    "source/deployment/cloudxr_teleoperation_cluster": "source/how-to/cloudxr_teleoperation",
    "source/deployment/cluster": "source/features/hpc_clusters",
    "source/deployment/docker": "source/isaaclab/installation",
    "source/deployment/index": "source/isaaclab/installation",
    "source/deployment/run_docker_example": "source/isaaclab/installation",
    "source/experimental-features/newton-physics-integration/visualization": "source/concepts/visualization",
    "source/how-to/draw_markers": "source/how-to/create_visualization_markers",
    "source/how-to/estimate_how_many_cameras_can_run": "source/how-to/simulation_performance",
    "source/how-to/optimize_stage_creation": "source/how-to/simulation_performance",
    "source/migration/comparing_simulation_isaacgym": "source/isaaclab/migrating",
    "source/migration/migrating_from_isaacgymenvs": "source/isaaclab/migrating",
    "source/migration/migrating_from_omniisaacgymenvs": "source/isaaclab/migrating",
    "source/migration/migrating_from_orbit": "source/isaaclab/migrating",
    "source/overview/core-concepts/actuators": "source/concepts/actuators",
    "source/overview/core-concepts/index": "source/concepts/task_workflows",
    "source/overview/core-concepts/motion_generators": "source/concepts/motion_generators",
    "source/overview/core-concepts/sensors/camera": "source/concepts/sensors/camera",
    "source/overview/core-concepts/sensors/contact_sensor": "source/concepts/sensors/contact_sensor",
    "source/overview/core-concepts/sensors/frame_transformer": "source/concepts/sensors/frame_transformer",
    "source/overview/core-concepts/sensors/imu": "source/concepts/sensors/imu",
    "source/overview/core-concepts/sensors/index": "source/concepts/sensors/index",
    "source/overview/core-concepts/sensors/ray_caster": "source/concepts/sensors/ray_caster",
    "source/overview/core-concepts/sensors/visuo_tactile_sensor": "source/experimental-features/visuo_tactile_sensor",
    "source/overview/core-concepts/task_workflows": "source/concepts/task_workflows",
    "source/overview/developer-guide/development": "source/refs/contributing",
    "source/overview/developer-guide/index": "source/refs/contributing",
    "source/overview/developer-guide/vs_code": "source/developer-tools/editor_setup",
    "source/overview/environments": "source/isaaclab/environments",
    "source/overview/imitation-learning/augmented_imitation": "source/features/imitation-learning/augmented_imitation",
    "source/overview/imitation-learning/index": "source/features/imitation-learning/index",
    "source/overview/imitation-learning/skillgen": "source/features/imitation-learning/skillgen",
    "source/overview/imitation-learning/teleop_imitation": "source/features/imitation-learning/teleop_imitation",
    "source/overview/own-project/index": "source/developer-tools/template_generator",
    "source/overview/own-project/project_structure": "source/developer-tools/template_generator",
    "source/overview/own-project/template": "source/developer-tools/template_generator",
    "source/overview/reinforcement-learning/index": "source/concepts/reinforcement_learning",
    "source/overview/reinforcement-learning/rl_existing_scripts": "source/how-to/run_rl_training",
    "source/overview/reinforcement-learning/rl_frameworks": "source/concepts/reinforcement_learning",
    "source/overview/reinforcement-learning/training_guide": "source/how-to/configuring_rl_training",
    "source/overview/showroom": "source/isaaclab/demos",
    "source/refs/migration": "source/isaaclab/migrating",
    "source/setup/installation/asset_caching": "source/how-to/manage_asset_downloads",
    "source/setup/installation/binaries_installation": "source/isaaclab/installation",
    "source/setup/installation/cloud_installation": "source/features/cloud_deployment",
    "source/setup/installation/isaaclab_pip_installation": "source/isaaclab/installation",
    "source/setup/installation/pip_installation": "source/isaaclab/installation",
    "source/setup/installation/source_installation": "source/isaaclab/installation",
}

# Pages without a current equivalent remain available in the final 2.x documentation release.
_isaaclab_v2_3_2_docs_root = "https://isaac-sim.github.io/IsaacLab/v2.3.2"
_isaaclab_v2_3_2_only_docs = (
    "source/experimental-features/newton-physics-integration/index",
    "source/experimental-features/newton-physics-integration/installation",
    "source/experimental-features/newton-physics-integration/isaaclab_newton-beta-2",
    "source/experimental-features/newton-physics-integration/limitations-and-known-bugs",
    "source/experimental-features/newton-physics-integration/sim-to-real",
    "source/experimental-features/newton-physics-integration/sim-to-sim",
    "source/experimental-features/newton-physics-integration/solver-transitioning",
    "source/experimental-features/newton-physics-integration/training-environments",
    "source/overview/developer-guide/repo_structure",
    "source/overview/reinforcement-learning/performance_benchmarks",
    "source/overview/simple_agents",
    "source/policy_deployment/00_hover/hover_policy",
    "source/policy_deployment/01_io_descriptors/io_descriptors_101",
    "source/refs/reference_architecture/index",
    "source/setup/installation/include/bin_verify_isaacsim",
    "source/setup/installation/include/pip_python_virtual_env",
    "source/setup/installation/include/pip_verify_isaacsim",
    "source/setup/installation/include/src_build_isaaclab",
    "source/setup/installation/include/src_clone_isaaclab",
    "source/setup/installation/include/src_python_virtual_env",
    "source/setup/installation/include/src_symlink_isaacsim",
    "source/setup/installation/include/src_verify_isaaclab",
    "source/setup/walkthrough/api_env_design",
    "source/setup/walkthrough/concepts_env_design",
    "source/setup/walkthrough/index",
    "source/setup/walkthrough/technical_env_design",
    "source/setup/walkthrough/training_jetbot_gt",
    "source/setup/walkthrough/training_jetbot_reward_exploration",
)
isaaclab_doc_redirects.update(
    {route: f"{_isaaclab_v2_3_2_docs_root}/{route}.html" for route in _isaaclab_v2_3_2_only_docs}
)

_streaming_how_to = "source/how-to/visualizer_streaming_camera_view"

# Sections of moved or consolidated pages and where they live now.
isaaclab_doc_redirect_fragments = {
    "source/concepts/schema_fragments": {
        "the-fragment-model": "source/concepts/schema_cfgs#add-backend-specific-settings",
        "targeting-expressions": "source/concepts/schema_cfgs#configure-an-existing-robot",
        "configuring-fragments-on-spawners": "source/concepts/schema_cfgs#add-backend-specific-settings",
        "creating-missing-apis": "source/concepts/schema_cfgs#when-a-setting-does-not-apply",
        "see-also": "source/concepts/schema_cfgs#schema-reference-and-implementation",
    },
    "source/concepts/backend_architecture": {
        "overview": "source/concepts/backends_and_presets#backend-architecture",
    },
    "source/concepts/joint_and_body_ordering": {
        "": "source/how-to/transfer_policies_between_physx_and_newton#joint-and-body-ordering",
        **{
            anchor: f"source/how-to/transfer_policies_between_physx_and_newton#{anchor}"
            for anchor in (
                "joint-and-body-ordering",
                "why-articulation-orders-differ",
                "public-and-backend-order",
                "high-level-mdp-terms",
                "conversion-cost",
                "direct-backend-view-access",
                "torch-conversion",
                "warp-conversion",
                "what-ordering-does-not-solve",
                "verification-and-troubleshooting",
            )
        },
    },
    "source/concepts/native-physics-api/newton": {
        "": "source/concepts/physics-backends/newton#newton-native-access",
        "newton-native-data-and-selection-api": "source/concepts/physics-backends/newton#newton-native-access",
    },
    "source/concepts/native-physics-api/ovphysx": {
        "": "source/concepts/physics-backends/ovphysx#ovphysx-native-access",
        "ovphysx-tensor-bindings": "source/concepts/physics-backends/ovphysx#ovphysx-native-access",
    },
    "source/concepts/native-physics-api/physx": {
        "": "source/concepts/physics-backends/physx#physx-native-access",
        "physx-tensor-api": "source/concepts/physics-backends/physx#physx-native-access",
    },
    "source/concepts/warp_environment_migration": {
        "": "source/concepts/warp_environments#warp-env-migration",
        **{
            anchor: f"source/concepts/warp_environments#{anchor}"
            for anchor in (
                "warp-env-migration",
                "design-rationale",
                "project-structure",
                "writing-warp-mdp-terms",
                "imports",
                "common-pattern",
                "observation-terms",
                "event-terms",
                "action-terms",
                "capture-safety",
                "parity-testing",
                "available-warp-mdp-terms",
            )
        },
    },
    "source/workflows/index": {
        "": "source/isaaclab/installation#installation-method-container",
    },
    "source/workflows/docker/index": {
        "": "source/isaaclab/installation#installation-method-container",
        "running-isaac-lab-in-docker": "source/isaaclab/installation#installation-method-container",
    },
    "source/workflows/docker/images": {
        "": "source/isaaclab/installation#docker-images",
    },
    "source/features/docker_cloud": {
        "": "source/isaaclab/installation#installation-method-container",
        "clusters": "source/features/hpc_clusters#deployment-cluster",
        "deployment-cluster": "source/features/hpc_clusters#deployment-cluster",
        "cloud-workstations": "source/features/cloud_deployment#docker-cloud-cloud",
        "docker-cloud-cloud": "source/features/cloud_deployment#docker-cloud-cloud",
    },
    "source/features/record_video": {
        "overview": "source/concepts/visualization#video-recording",
        "record-video-cli": "source/concepts/visualization#record-video-cli",
        "usage": "source/how-to/record_video#sources",
        "source-types": "source/how-to/record_video#sources",
        "limitations-and-compatibility": "source/how-to/record_video#limitations",
    },
    "source/features/visualizer_tiled_camera": {
        "overview": "source/concepts/visualization#visualization-streaming-camera-view",
        "usage": "source/concepts/visualization#visualization-streaming-camera-view",
        "configuration-notes": f"{_streaming_how_to}#configuration",
        "migration-from-generated-streaming-cameras": f"{_streaming_how_to}#migrating-from-generated-cameras",
        "declared-display-sources": f"{_streaming_how_to}#display-sources",
    },
}

# -- Advanced configuration -------------------------------------------------


def skip_member(app, what, name, obj, skip, options):
    # List the names of the functions you want to skip here
    exclusions = ["from_dict", "to_dict", "replace", "copy", "validate", "__post_init__"]
    if name in exclusions:
        return True
    return None


def update_environment_browser(app):
    """Regenerate the environment browser's task rows from the task registry, so they cannot go stale."""
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "update_environments_rst.py")
    subprocess.run([sys.executable, tool], check=True)


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    app.connect("builder-inited", update_environment_browser)

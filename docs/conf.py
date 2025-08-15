# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
import sys

sys.path.insert(0, os.path.abspath("../source/isaaclab"))
sys.path.insert(0, os.path.abspath("../source/isaaclab/isaaclab"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_tasks"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_tasks/isaaclab_tasks"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_rl"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_rl/isaaclab_rl"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_mimic"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_mimic/isaaclab_mimic"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_assets"))
sys.path.insert(0, os.path.abspath("../source/isaaclab_assets/isaaclab_assets"))

# -- Project information -----------------------------------------------------

project = "Isaac Lab"
copyright = "2022-2025, The Isaac Lab Project Developers."
author = "The Isaac Lab Project Developers."

# Read version from the package
with open(os.path.join(os.path.dirname(__file__), "..", "VERSION")) as f:
    full_version = f.read().strip()
    version = ".".join(full_version.split(".")[:3])

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
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.icon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tabs.tabs",  # backwards compatibility for building docs on v1.0.0
    "sphinx_multiversion",
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
    "torch": ("https://pytorch.org/docs/stable/", None),
    "isaac": ("https://docs.omniverse.nvidia.com/py/isaacsim", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "warp": ("https://nvidia.github.io/warp/", None),
    "dev-guide": ("https://docs.omniverse.nvidia.com/dev-guide/latest", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_redirect", "_templates", "Thumbs.db", ".DS_Store", "README.md", "licenses/*"]

# Mock out modules that are not available on RTD
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "numpy",
    "matplotlib",
    "scipy",
    "carb",
    "warp",
    "pxr",
    "isaacsim",
    "omni",
    "omni.kit",
    "omni.log",
    "omni.usd",
    "omni.client",
    "omni.physx",
    "omni.physics",
    "pxr.PhysxSchema",
    "pxr.PhysicsSchemaTools",
    "omni.replicator",
    "omni.isaac.core",
    "omni.isaac.kit",
    "omni.isaac.cloner",
    "omni.isaac.urdf",
    "omni.isaac.version",
    "omni.isaac.motion_generation",
    "omni.isaac.ui",
    "isaacsim",
    "isaacsim.core.api",
    "isaacsim.core.cloner",
    "isaacsim.core.version",
    "isaacsim.robot_motion.motion_generation",
    "isaacsim.gui.components",
    "isaacsim.asset.importer.urdf",
    "isaacsim.asset.importer.mjcf",
    "omni.syntheticdata",
    "omni.timeline",
    "omni.ui",
    "gym",
    "skrl",
    "stable_baselines3",
    "rsl_rl",
    "rl_games",
    "ray",
    "h5py",
    "hid",
    "prettytable",
    "tqdm",
    "tensordict",
    "trimesh",
    "toml",
    "pink",
    "pinocchio",
    "nvidia.srl",
    "flatdict",
    "IPython",
    "ipywidgets",
    "mpl_toolkits",
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
html_static_path = ["source/_static/css"]
html_css_files = ["custom.css"]

html_theme_options = {
    "path_to_docs": "docs/",
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
            "icon": "https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg",
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
smv_branch_whitelist = os.getenv("SMV_BRANCH_WHITELIST", r"^(main|devel)$")
# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = os.getenv("SMV_TAG_WHITELIST", r"^v[1-9]\d*\.\d+\.\d+$")
html_sidebars = {
    "**": ["navbar-logo.html", "versioning.html", "icon-links.html", "search-field.html", "sbt-sidebar-nav.html"]
}


# -- Advanced configuration -------------------------------------------------


def skip_member(app, what, name, obj, skip, options):
    # List the names of the functions you want to skip here
    exclusions = ["from_dict", "to_dict", "replace", "copy", "validate", "__post_init__"]
    if name in exclusions:
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", skip_member)

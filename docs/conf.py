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

sys.path.insert(0, os.path.abspath("../source/extensions/omni.isaac.orbit"))
sys.path.insert(0, os.path.abspath("../source/extensions/omni.isaac.orbit/omni/isaac/orbit"))
sys.path.insert(0, os.path.abspath("../source/extensions/omni.isaac.orbit_envs"))
sys.path.insert(0, os.path.abspath("../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs"))

# -- Project information -----------------------------------------------------

project = "orbit"
copyright = "2022, NVIDIA, ETH Zurich and University of Toronto"
author = "NVIDIA, ETH Zurich and University of Toronto"

version = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "sphinxcontrib.bibtex",
    "myst_parser",
    "autodocsumm",
    "sphinx_copybutton",
    "sphinx_panels",
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

# put type hints inside the description instead of the signature (easier to read)
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# document class *and* __init__ methods
autoclass_content = "class"  #
# separate class docstring from __init__ docstring
autodoc_class_signature = "separated"
# sort members by source order
autodoc_member_order = "groupwise"
# BibTeX configuration
bibtex_bibfiles = ["source/_static/refs.bib"]
# default autodoc settings
autodoc_default_options = {
    "autosummary": True,
}

# generate links to the documentation of objects in external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "licenses/*"]

# Mock out modules that are not available on RTD
autodoc_mock_imports = [
    "torch",
    "numpy",
    "matplotlib",
    "scipy",
    "carb",
    "warp",
    "pxr",
    "omni.kit",
    "omni.usd",
    "omni.client",
    "pxr.PhysxSchema",
    "pxr.PhysicsSchemaTools",
    "omni.replicator",
    "omni.isaac.core",
    "omni.isaac.kit",
    "omni.isaac.cloner",
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
    "toml",
]

# List of zero or more Sphinx-specific warning categories to be squelched (i.e.,
# suppressed, ignored).
suppress_warnings = [
    # FIXME: *THIS IS TERRIBLE.* Generally speaking, we do want Sphinx to inform
    # us about cross-referencing failures. Remove this hack entirely after Sphinx
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

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_book_theme

html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_logo = "source/_static/nv-logo.png"
html_favicon = "source/_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["source/_static"]
html_css_files = ["css/nvidia.css"]

html_theme_options = {
    "collapse_navigation": True,
    "repository_url": "https://github.com/NVIDIA-Omniverse/Orbit",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "use_sidenotes": True,
    "announcement": "⚠️This is a pre-release version of Orbit. Please report any issues on <a href='https://github.com/NVIDIA-Omniverse/orbit/issues'>GitHub</a>.",
}

html_show_copyright = True
html_show_sphinx = False

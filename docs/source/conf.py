import os
import sys
from datetime import datetime

# Make your package importable for autodoc
sys.path.insert(0, os.path.abspath("../.."))

project = "qkerasV3"
author = "qkerasV3 contributors"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Automatically generate autosummary pages
autosummary_generate = True

# Docstring parsing
# The repository uses docstrings that look closest to Google style ("Returns:", etc.)
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Show type hints in the description (avoids overly noisy signatures)
autodoc_typehints = "description"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

# -- MyST / notebook configuration ---------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Do not execute notebooks during documentation builds
nb_execution_mode = "off"

# -- Keras / TensorFlow configuration ------------------------------------

# qkerasV3 currently supports TensorFlow backend via Keras 3.
# Setting this here makes autodoc imports stable in CI/RTD.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# -- Options for HTML output ---------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

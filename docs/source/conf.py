import os
import sys

# Make your package importable for autodoc
sys.path.insert(0, os.path.abspath("../.."))

project = "qkeras-v3"
author = "qkerasV3 contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",  # for Markdown pages
    # Optional: render notebooks (recommended if you want notebooks/ in docs)
    # "myst_nb",
]

autosummary_generate = True

# Your docstrings look closer to Google-style ("Returns:" etc.)
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "sphinx_rtd_theme"

# Keras 3 backend selection for doc builds
os.environ["KERAS_BACKEND"] = "tensorflow"

# Helps avoid TF trying to allocate GPU stuff in CI (usually not needed, but safe)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Optional: if importing qkeras triggers heavy imports, you can mock modules here:
# autodoc_mock_imports = ["tensorflow"]

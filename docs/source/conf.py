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

import pyDeltaRCM

# -- Project information -----------------------------------------------------

project = 'pyDeltaRCM'
copyright = '2020, The DeltaRCM Team'
author = 'The DeltaRCM Team'

# The full version, including alpha/beta/rc tags
release = pyDeltaRCM.__version__
version = pyDeltaRCM.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.graphviz',
              'sphinx.ext.imgmath',
              'sphinx.ext.githubpages',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.todo']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# toggle todo items
todo_include_todos = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autosummary / Automodapi settings
autosummary_generate = True
automodapi_inheritance_diagram = False
autodoc_default_options = {'members': True, 'inherited-members': False,
                           'private-members': True}

# doctest
doctest_global_setup = '''
import pyDeltaRCM
import numpy as np
from matplotlib import pyplot as plt
'''
doctest_test_doctest_blocks = ''  # empty string disables testing all code in any docstring

## mpl plots
plot_basedir = 'pyplots'
plot_html_show_source_link = False
plot_formats = ['png', ('hires.png', 300)]
plot_pre_code = '''
import numpy as np
from matplotlib import pyplot as plt
import pyDeltaRCM
'''


# img math
# imgmath_latex_preamble = '\\usepackage{fouriernc}' # newtxsf, mathptmx

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_static_path = []

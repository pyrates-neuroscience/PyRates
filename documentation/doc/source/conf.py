# -*- coding: utf-8 -*-

import sys
import os
import re

from sphinx.locale import _
from pyrates import __version__ as version

# Prefer to use the version of the theme in this repo
# and not the installed version of the theme.
sys.path.insert(0, os.path.abspath('../../'))
sys.path.append(os.path.abspath('sphinxext'))

project = u'PyRates Documentation'
slug = re.sub(r'\W+', '-', project.lower())
version = ".".join(version.split('.')[:-1])
release = version
author = u'Richard Gast and Daniel Rose'
copyright = author
language = 'en'

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinxcontrib.httpdomain',
    'sphinx_rtd_theme',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.extlinks',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_gallery.gen_gallery'
]

# configuration of sphinx gallery
sphinx_gallery_conf = {
    'examples_dirs': [
        '../../model_introductions',
        '../../model_implementation',
        '../../model_analysis',
    ],   # path to your example scripts
    'gallery_dirs': [
        'auto_introductions',
        'auto_implementations',
        'auto_analysis',
    ]  # path to where to save gallery generated output

}

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

templates_path = ['_templates']
source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
locale_dirs = ['locale/']
gettext_compact = False

master_doc = 'index'
suppress_warnings = ['image.nonlocal_uri']
pygments_style = 'default'

intersphinx_mapping = {
    'rtd': ('https://docs.readthedocs.io/en/stable/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'navigation_depth': 3
}
html_context = {}
html_static_path = ['_static']
html_short_title = 'PyRates'
html_logo = '_static/img/pyrates_logo.png'
html_show_sourcelink = True
htmlhelp_basename = slug

pngmath_latex_preamble = r"""
\usepackage{color}
\definecolor{textgray}{RGB}{51,51,51}
\color{textgray}
"""
pngmath_use_preview = True
pngmath_dvipng_args = ['-gamma 1.5', '-D 96', '-bg Transparent']

latex_documents = [
  ('index', '{0}.tex'.format(slug), project, author, 'manual'),
]

man_pages = [
    ('index', slug, project, [author], 1)
]

texinfo_documents = [
  ('index', slug, project, author, slug, project, 'Miscellaneous'),
]


# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    'extraclassoptions': 'openany,oneside',
    'babel': '\\usepackage[shorthands=off]{babel}'
}


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# Extensions to theme docs
def setup(app):
    from sphinx.domains.python import PyField
    from sphinx.util.docfields import Field

    app.add_object_type(
        'confval',
        'confval',
        objname='configuration value',
        indextemplate='pair: %s; configuration value',
        doc_field_types=[
            PyField(
                'type',
                label=_('Type'),
                has_arg=False,
                names=('type',),
                bodyrolename='class'
            ),
            Field(
                'default',
                label=_('Default'),
                has_arg=False,
                names=('default',),
            ),
        ]
    )

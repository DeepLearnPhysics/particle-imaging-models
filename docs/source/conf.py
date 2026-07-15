# Configuration file for the Sphinx documentation builder.
#
# pimm - Particle Imaging Models documentation.
# Built with Sphinx + MyST (Markdown) + PyData Sphinx Theme. The top-level
# sections live in the horizontal header; page-local navigation stays out of
# the reading column until it is needed on smaller screens.
# Narrative guides need only the doc deps, but the API reference uses autodoc
# (and gen_api.py walks the live registries), so a full build imports `pimm` and
# must run in the project environment. See docs/DEPLOYMENT.md.

import datetime
import os

# -- Project information -----------------------------------------------------

project = "pimm"
author = "Samuel Young"
copyright = f"{datetime.date.today().year}, DeepLearnPhysics"

# The documentation describes the current repository rather than advertising a
# package release in the site chrome. Release history remains available from
# GitHub; keeping these blank prevents Sphinx themes from adding a stale version
# label to the title or sidebar.
release = ""
version = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    # API reference (autodoc imports pimm - the build env has the full stack).
    # Mirrors torchrl's reference stack.
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

# sphinx-sitemap starts a multiprocessing manager during builder setup. Some
# restricted documentation-preview environments forbid local sockets; they can
# skip only sitemap generation while keeping the rendered HTML identical.
if os.environ.get("PIMM_DOCS_DISABLE_SITEMAP") != "1":
    extensions.append("sphinx_sitemap")

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_class_signature = "mixed"
# Do not inherit PyTorch's generic ``Module.forward`` prose onto pimm methods
# that intentionally document only their own signature. The universal calling
# rule (``module(...)``, not ``module.forward(...)``) is explained once in the
# API landing page instead of repeated on every generated class page.
autodoc_inherit_docstrings = False
add_module_names = False  # show `from_pretrained`, not `pimm.export.from_pretrained`
python_use_unqualified_type_names = True

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
    "inherited-members": False,
}

# Optional/heavy deps that may be absent on a docs-only host. The project env has
# the full stack, but mocking these keeps the build resilient on a thin host.
autodoc_mock_imports = [
    "spconv",
    "ocnn",
    "MinkowskiEngine",
    "pointops",
    "pointgroup_ops",
    "cnms",
    "pytorch3d_ops",
    "pointrope",
    "serialize_cuda",
    "torch_scatter",
    "torch_geometric",
    "torch_cluster",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

# Narrative pages are Markdown; the API reference is reStructuredText. List
# ``.rst`` FIRST so Sphinx's autosummary writes its generated class stubs as
# ``.rst`` (its content is rst). If ``.md`` is the default suffix, autosummary
# emits ``.md`` stubs full of rst directives, which MyST renders as literal text.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Avoid duplicate-label noise from autosectionlabel across many pages.
autosectionlabel_prefix_document = True
suppress_warnings = ["autosectionlabel.*"]

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "attrs_inline",
    "attrs_block",
    "substitution",
    "tasklist",
    "linkify",
    "smartquotes",
]
myst_heading_anchors = 3
myst_substitutions = {
    "repo_url": "https://github.com/DeepLearnPhysics/particle-imaging-models",
    "docker_image": "youngsm/pimm:pytorch2.10.0-cuda12.6",
}

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}
intersphinx_disabled_reftypes = ["*"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "pimm"
html_static_path = ["_static"]
html_css_files = ["custom.css", "launch_selector.css"]
html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
    "launch_selector.js",
]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

# Published base URL (used by sphinx-sitemap and canonical links).
html_baseurl = "https://deeplearnphysics.org/particle-imaging-models/stable/"
sitemap_url_scheme = "{link}"

html_theme_options = {
    "logo": {
        "text": "pimm",
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
        "alt_text": "pimm - Particle Imaging Models",
    },
    "use_edit_page_button": False,
    "show_prev_next": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "header_links_before_dropdown": 5,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/DeepLearnPhysics/particle-imaging-models",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Hugging Face",
            "url": "https://huggingface.co/DeepLearnPhysics",
            # fa-hugging-face only exists in Font Awesome 7.2+, but the theme
            # bundle on the deployed site can lag behind - register our own copy.
            "icon": "fa-custom fa-huggingface",
            "type": "fontawesome",
        },
        {
            "name": "DeepLearnPhysics",
            "url": "https://deeplearnphysics.org",
            "icon": "fa-solid fa-globe",
        },
    ],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "announcement": (
        "pimm is research software under active development - "
        "APIs and configs may change between versions."
    ),
}

html_context = {
    "github_user": "DeepLearnPhysics",
    "github_repo": "particle-imaging-models",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
}

# Copybutton: ignore shell prompts and pseudo-prompts in code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

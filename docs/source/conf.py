# Configuration file for the Sphinx documentation builder.
#
# pimm - Particle Imaging Models documentation.
# Built with Sphinx + MyST (Markdown) + sphinx-book-theme. The book theme is
# PyData-based (so the --pst-* CSS variables apply) and renders the FULL nested,
# collapsible section tree in the left sidebar on every page - every section is
# reachable from anywhere, which pytorch_sphinx_theme2 / plain PyData cannot do
# (they scope the left sidebar to the current section).
# Narrative guides need only the doc deps, but the API reference uses autodoc
# (and gen_api.py walks the live registries), so a full build imports `pimm` and
# must run in the project environment. See docs/DEPLOYMENT.md.

import datetime

# -- Project information -----------------------------------------------------

project = "pimm"
author = "Samuel Young"
copyright = f"{datetime.date.today().year}, DeepLearnPhysics"

# The full version, including alpha/beta/rc tags. Keep in sync with
# pyproject.toml. The docs site is published under
# /particle-imaging-models/stable/.
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_sitemap",
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

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_class_signature = "mixed"
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
    ".rst": "restructuredtext",
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
    "colon_fence",
    "deflist",
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

html_theme = "sphinx_book_theme"
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
    "repository_url": "https://github.com/DeepLearnPhysics/particle-imaging-models",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "home_page_in_toc": True,
    # Left sidebar: render the whole tree, expanded a couple of levels, deep
    # enough to reach every page.
    "show_navbar_depth": 2,
    "max_navbar_depth": 4,
    "collapse_navbar": False,
    "show_toc_level": 2,
    "navigation_with_keys": False,
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

# Show the full navigation sidebar on every page, including the landing page,
# so the section tree is always available (no per-section disconnection).

# Copybutton: ignore shell prompts and pseudo-prompts in code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

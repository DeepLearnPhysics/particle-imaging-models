# Sphinx configuration for the pimm documentation.
#
# Pages are MyST Markdown rendered with the PyData Sphinx Theme. The Reference
# pages for the command line, recipes, Python API and environment variables are
# generated from the code by docs/tools/gen_reference.py, which `make html` runs
# before Sphinx.

import datetime
import os
import re
from importlib import metadata
from pathlib import Path

project = "pimm"
author = "Samuel Young"
copyright = f"{datetime.date.today().year}, DeepLearnPhysics"

try:
    release = metadata.version("pimm")
except metadata.PackageNotFoundError:
    _pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    release = re.search(r'^version = "([^"]+)"', _pyproject, flags=re.M).group(1)
version = release

extensions = ["myst_parser", "sphinx_copybutton"]
# sphinx-sitemap starts a multiprocessing manager; environments that forbid
# local sockets can skip it without changing the rendered pages.
if os.environ.get("PIMM_DOCS_DISABLE_SITEMAP") != "1":
    extensions.append("sphinx_sitemap")

source_suffix = {".md": "markdown"}
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"version": release}

# Copy code exactly as shown. Only interactive prompts are stripped; "# " is a
# comment, not a prompt.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

html_theme = "pydata_sphinx_theme"
html_title = f"pimm {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
    ("pimm.js", {"defer": "defer"}),
    # Cronitor RUM
    ("https://rum.cronitor.io/script.js", {"async": "async"}),
    (
        None,
        {
            "body": "window.cronitor = window.cronitor || function() { "
            "(window.cronitor.q = window.cronitor.q || []).push(arguments); };\n"
            "cronitor('config', { clientKey: '7d1e937942c91525bb5b9db25632aa4c' });"
        },
    ),
]
html_logo = "_static/logo-light.svg"
html_favicon = "_static/logo-light.svg"

html_baseurl = "https://deeplearnphysics.org/particle-imaging-models/stable/"
sitemap_url_scheme = "{link}"

html_theme_options = {
    "logo": {
        "text": f"pimm {release}",
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
        "alt_text": "pimm",
    },
    "use_edit_page_button": False,
    "show_prev_next": True,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "header_links_before_dropdown": 6,
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
            "icon": "fa-custom fa-huggingface",
            "type": "fontawesome",
        },
    ],
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
}

html_context = {
    "github_user": "DeepLearnPhysics",
    "github_repo": "particle-imaging-models",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
}

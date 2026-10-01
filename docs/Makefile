# Sphinx documentation for pimm.
#
#   make html       generate the Reference pages from the code, then build with -W
#   make gen        regenerate the Reference pages only
#   make serve      build, then serve at http://localhost:8000
#   make linkcheck  check external links
#   make clean      remove the build and the generated pages

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PORT          ?= 8000
PYTHON        ?= python
GENERATED     = $(SOURCEDIR)/reference/cli.md $(SOURCEDIR)/reference/recipes.md \
                $(SOURCEDIR)/reference/api.md $(SOURCEDIR)/reference/environment.md

.PHONY: help gen html clean serve linkcheck Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

gen:
	@$(PYTHON) tools/gen_reference.py .. $(SOURCEDIR)

html: gen
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Build finished. Open $(BUILDDIR)/html/index.html"

clean:
	rm -rf "$(BUILDDIR)" $(GENERATED)

serve: html
	@echo "Serving on http://localhost:$(PORT)"
	@cd "$(BUILDDIR)/html" && python -m http.server $(PORT) --bind 127.0.0.1

linkcheck: gen
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

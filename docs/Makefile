# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: multi-docs
multi-docs:
	@echo "Generating version switcher..."
	@python3 generate_switcher.py
	@echo "Building multi-version documentation..."
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "Copying redirect page..."
	@cp _redirect/index.html $(BUILDDIR)/index.html
	@echo "Documentation built successfully!"

.PHONY: current-docs
current-docs:
	@rm -rf "$(BUILDDIR)/current"
	@$(SPHINXBUILD) -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS)

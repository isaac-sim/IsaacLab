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
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)" --jobs=auto $(SPHINXOPTS)
	@cp _redirect/index.html $(BUILDDIR)/index.html

.PHONY: current-docs
current-docs:
	@rm -rf "$(BUILDDIR)/current"
	@$(SPHINXBUILD) -W --keep-going -j auto "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS)

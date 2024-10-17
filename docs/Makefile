# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-multiversion
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: multi-docs
multi-docs:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@cp _redirect/index.html $(BUILDDIR)/index.html

.PHONY: single-docs
single-docs:
	@sphinx-build "$(SOURCEDIR)" "$(BUILDDIR)/latest" $(SPHINXOPTS)

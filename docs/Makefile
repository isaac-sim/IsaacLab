# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DOCS_DEFAULT_REF ?= main

.PHONY: multi-docs
multi-docs:
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@test -f "$(BUILDDIR)/$(DOCS_DEFAULT_REF)/index.html" || \
		(echo "Default docs ref '$(DOCS_DEFAULT_REF)' was not built" && exit 1)
	@sed 's|__DOCS_DEFAULT_REF__|$(DOCS_DEFAULT_REF)|g' _redirect/index.html > $(BUILDDIR)/index.html

.PHONY: current-docs
current-docs:
	@rm -rf "$(BUILDDIR)/current"
	@$(SPHINXBUILD) -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS)

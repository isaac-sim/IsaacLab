# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DOCS_DEFAULT_REF ?= v3.0.0-EA

.PHONY: multi-docs
multi-docs:
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)" --jobs=auto $(SPHINXOPTS)
	@test -f "$(BUILDDIR)/$(DOCS_DEFAULT_REF)/index.html" || \
		(echo "Default docs ref '$(DOCS_DEFAULT_REF)' was not built" && exit 1)
	@sed 's|url=\./[^"]*/index.html|url=./$(DOCS_DEFAULT_REF)/index.html|' _redirect/index.html > $(BUILDDIR)/index.html

.PHONY: current-docs
current-docs:
	@rm -rf "$(BUILDDIR)/current"
	@$(SPHINXBUILD) -W --keep-going -j auto "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS)

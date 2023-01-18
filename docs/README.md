# Building Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) with the [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/) for maintaining the documentation.

> **Note:** To build the documentation, we recommend creating a virtual environment to avoid any conflicts with system installed dependencies.

Execute the following instructions to build the documentation (assumed from the top of the repository):

1. Install the dependencies for [Sphinx](https://www.sphinx-doc.org/en/master/):

    ```bash
    # enter the location where this readme exists
    cd docs
    # install dependencies
    pip install -r requirements.txt
    ```

2. Generate the documentation file via:

    ```bash
    # make the html version
    make html
    ```

3. The documentation is now available at `docs/_build/html/index.html`:

    ```bash
    # open on default browser
    xdg-open _build/html/index.html
    ```

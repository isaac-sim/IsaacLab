* Updated Isaac Lab to use Isaac Sim 6.2, including the pip wheel, standalone importer,
  container image, and China storage profile.
* Raised the ``aiohttp`` and ``urllib3`` dependency floors to versions that address
  GHSA-cq5v-8q36-5273 and GHSA-vxq7-64xx-v4gw.
* Raised the ``GitPython`` security floor to 3.1.60 and refreshed the locked ``fsspec``
  and ``hydra-core`` releases used by the default environment.

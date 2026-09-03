# ecr-build-push-pull

Builds a Docker image and pushes it to ECR, or pulls it if the tag already exists.
ECR is also used as the BuildKit layer cache.

## Usage

```yaml
- uses: ./.github/actions/ecr-build-push-pull
  with:
    image-tag: ${{ env.DOCKER_IMAGE_TAG }}
    isaacsim-base-image: nvcr.io/nvidia/isaac-sim
    isaacsim-version: 6.0.0
    dockerfile-path: docker/Dockerfile.base
    cache-tag: cache-base
    ecr-url: (optional, complete url for ECR storage)
```

## Verifying a freshly built image

Pass `verify-test-path` to assert against the image before it is published:

```yaml
    verify-test-path: docker/test/test_image_invariants.py
```

The tests run only on a full build, with `IMAGE_TAG` set, so the caller's job needs `uv`
(`astral-sh/setup-uv`). A failure fails the action with nothing pushed, so the next run
rebuilds rather than serving the bad image from the deps cache. Exact-tag and deps-cache hits skip
them: that image passed when it was built.

## ECR URL resolution order

1. `ecr-url` input
2. `ECR_CACHE_URL` environment variable on the runner
3. SSM parameter `/github-runner/<instance-id>/ecr-cache-url`
4. If none resolve, ECR is skipped and the image is built locally

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

## Outputs

- `built`: `'true'` when this run built the image, so it exists locally under `image-tag`.
  An exact-tag hit pulls it as well; a deps-cache hit only creates the ECR tag unless
  `pull-on-deps-hit` is set. Gate steps that need the image locally on this output.

## ECR URL resolution order

1. `ecr-url` input
2. `ECR_CACHE_URL` environment variable on the runner
3. SSM parameter `/github-runner/<instance-id>/ecr-cache-url`
4. If none resolve, ECR is skipped and the image is built locally

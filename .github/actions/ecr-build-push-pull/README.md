# ecr-build-push-pull

Action that ether builds Docker image and pushes it to ECR or pulls it from there.

ECR is also used as the BuildKit layer cache, so even first builds benefit from cached layers.

Drop-in replacement for `docker-build/action.yml` with ECR-backed caching.

## Prerequisites

The runner must be authenticated to AWS; the action calls
`aws ecr get-login-password` using whatever credentials are available in the environment.

The runner must also have the `ECR_CACHE_URL` environment variable set to the full ECR
repository URL (e.g. `123456789.dkr.ecr.us-west-2.amazonaws.com/my-repo`).

The IAM role must have at minimum:
- `ecr:GetAuthorizationToken` (on `*`)
- `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage` (pull)
- `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:PutImage` (push)

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `image-tag` | yes | — | Local Docker tag (e.g. `isaac-lab-dev:pr-1-abc`) |
| `isaacsim-base-image` | yes | — | IsaacSim base image (`ISAACSIM_BASE_IMAGE_ARG` build-arg) |
| `isaacsim-version` | yes | — | IsaacSim version (`ISAACSIM_VERSION_ARG` build-arg) |
| `dockerfile-path` | no | `docker/Dockerfile.base` | Path to Dockerfile |
| `ecr-url` | no | `""` | Full ECR repository URL. If omitted, falls back to the `ECR_CACHE_URL` env var on the runner. If neither is set, ECR push/pull/cache is skipped. |
| `cache-tag` | no | `cache` | Tag used for the ECR layer cache image. |

## Usage

```yaml
- uses: ./.github/actions/ecr-build-push-pull
  with:
    image-tag: ${{ env.DOCKER_IMAGE_TAG }}
    isaacsim-base-image: nvcr.io/nvidia/isaac-sim
    isaacsim-version: 6.0.0
    dockerfile-path: docker/Dockerfile.base
    cache-tag: cache-base
    # ecr-url is optional — the runner's ECR_CACHE_URL env var is used automatically
```

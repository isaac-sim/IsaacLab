# ecr-build-push-pull

Action that ether builds Docker image and pushes it to ECR or pulls it from there.

ECR is also used as the BuildKit layer cache, so even first builds benefit from cached layers.

Drop-in replacement for `docker-build/action.yml` with ECR-backed caching.

## Prerequisites

The runner must be authenticated to AWS; the action calls
`aws sts get-caller-identity` and `aws ecr get-login-password` using whatever credentials
are available in the environment.

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
| `ecr-repository` | no | `""` | ECR repo name. If omitted, ECR push/pull/cache is skipped. |
| `aws-region` | no | `us-west-2` | ECR region. Should match the runner's EC2 region for best performance. |

## Usage

```yaml
- uses: ./.github/actions/ecr-build-push-pull
  with:
    image-tag: ${{ env.DOCKER_IMAGE_TAG }}
    isaacsim-base-image: nvcr.io/nvidia/isaac-sim
    isaacsim-version: 6.0.0
    dockerfile-path: docker/Dockerfile.base
    ecr-repository: isaaclab-ci
    aws-region: us-west-2
```

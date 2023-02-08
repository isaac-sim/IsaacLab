# Running Docker Container
If you wish to run a container with a command that starts a GUI application you need to first manage X-server authentication. In order to authenticate the docker container alone the following command needs to be run to populate a temporary file which our docker build will use:

```
export DOCKER_XAUTH=/tmp/.docker.xauth
touch $DOCKER_XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $DOCKER_XAUTH nmerge -
```

One can build the container by running:

```
docker-compose build
```

One can run an application by running

```
docker-compose up
```

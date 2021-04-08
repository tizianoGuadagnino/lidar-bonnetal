#!/bin/bash
# This file is covered by the LICENSE file in the root of this project.
nvidia-docker build -t pcw-net .
nvidia-docker run --privileged \
       -ti --rm -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       --net=host \
       -v $1:/home/developer/data/ \
       pcw-net

#!/bin/bash
REPONAME="nikwl/tsp"
MY_USERNAME=`whoami`

# Lets you use the display 
xhost +local:$MY_USERNAME

sudo docker run \
    -it \
    --network host \
    --env DISPLAY=$DISPLAY \
    --env QT_X11_NO_MITSHM=1 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume `pwd`/..:/opt/ws \
    --volume $DATADIR:/opt/data \
    $REPONAME \
    bash
    
# Be sure to revoke the privilege 
xhost -local:$MY_USERNAME
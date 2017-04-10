#!/bin/bash
NV_GPU='1' nvidia-docker run -itd --net=host --name $1 -v `pwd`:/home th --port=$2

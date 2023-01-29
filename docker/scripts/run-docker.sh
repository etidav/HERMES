#!/bin/bash

#
# Script to initialize a new container based on the HMM_with_external_signals docker
#

container_name="hermes"
image_name="hermes"

cmd="docker run -d -it --rm --name ${container_name} --entrypoint bash ${image_name}"

echo $cmd
$cmd

echo "Your container is '$container_name', running image '$image_name'"

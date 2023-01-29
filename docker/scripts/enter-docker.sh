#!/bin/bash

#
# Script to enter a previously started container hmm_with_external_signals
#

container_name="hermes"

IP_DOCKER=$(docker ps | grep -w $container_name | cut -d' ' -f1)
echo "The unique ID of your docker is: $IP_DOCKER"
cmd="docker exec -it ${IP_DOCKER} bash"

echo "will now execute:"
echo $cmd
$cmd

true

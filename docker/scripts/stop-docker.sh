#!/bin/bash

#
# Script to enter a previously started container running datascience_stable (when called without argument) or running datascience_testing (when called with an argument)
#

container_name="hermes"

cmd="docker rm -f ${container_name}"

echo "will now execute:"
echo $cmd
$cmd

true

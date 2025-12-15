#!/bin/bash

# copy the ../pipeline_rl_results/$EXPERIMENT_NAME/log.txt to local
SSH_OPTS="-J tyhua@scdt.stanford.edu -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ConnectTimeout=10"

# Get the most recently modified experiment directory from remote server
EXPERIMENT_NAME=$(ssh $SSH_OPTS tyhua@node1-ccn2cluster.stanford.edu "ls -t /ccn2/u/tyhua/pipeline_rl_results/ | head -1")



echo "Using experiment: $EXPERIMENT_NAME"
# Create local directory if it doesn't exist
LOCAL_DIR=./results/$EXPERIMENT_NAME
mkdir -p $LOCAL_DIR
scp $SSH_OPTS tyhua@node1-ccn2cluster.stanford.edu:/ccn2/u/tyhua/pipeline_rl_results/$EXPERIMENT_NAME/log.txt $LOCAL_DIR/log.txt

cat $LOCAL_DIR/log.txt
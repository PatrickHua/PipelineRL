#!/bin/bash

# Get node number from argument, default to 1
NODE=${1:-1}

# SSH options for better connection handling
SSH_OPTS="-J tyhua@scdt.stanford.edu -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ConnectTimeout=10"

# Sync files
# --delete removes files on destination that don't exist on source
rsync -avz --delete -e "ssh $SSH_OPTS" --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'results' /Users/tianyu/Work/PipelineRL/ tyhua@node${NODE}-ccn2cluster.stanford.edu:/ccn2/u/tyhua/PipelineRL/

# Only SSH if rsync succeeded
if [ $? -eq 0 ]; then
    # Small delay to let connection stabilize
    sleep 1
    ssh $SSH_OPTS tyhua@node${NODE}-ccn2cluster.stanford.edu -t "cd /ccn2/u/tyhua/PipelineRL && tmux new -A -t pipeline-rl"
else
    echo "Rsync failed, not connecting via SSH"
    exit 1
fi
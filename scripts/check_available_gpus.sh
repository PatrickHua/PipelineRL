#!/bin/bash

# Function to check GPUs on a specific node
check_node_gpus() {
    local node=$1
    echo "=========================================="
    echo "Checking GPUs on node${node}-ccn2cluster.stanford.edu"
    echo "=========================================="
    
    # Get GPU info, filter out nvm warnings, and format it nicely
    local output=$(ssh -J tyhua@scdt.stanford.edu tyhua@node${node}-ccn2cluster.stanford.edu "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits" 2>&1 | grep -E "^[0-9]")
    
    if [ -z "$output" ]; then
        echo "Failed to check node ${node} or no GPUs found"
        echo ""
        return
    fi
    
    echo "$output" | while IFS=',' read -r index name mem_used mem_total gpu_util mem_util; do
        # Trim whitespace
        index=$(echo "$index" | xargs)
        name=$(echo "$name" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)
        mem_util=$(echo "$mem_util" | xargs)
        
        # Convert MB to GB for display
        mem_used_gb=$(awk "BEGIN {printf \"%.1f\", $mem_used/1024}")
        mem_total_gb=$(awk "BEGIN {printf \"%.1f\", $mem_total/1024}")
        
        # Calculate memory percentage
        mem_percent=$(awk "BEGIN {printf \"%.0f\", ($mem_used/$mem_total)*100}")
        mem_free_percent=$((100 - mem_percent))
        
        # Determine availability status
        if [ "$gpu_util" -lt 5 ] && [ "$mem_percent" -lt 10 ]; then
            status="🟢 Available"
        elif [ "$gpu_util" -lt 50 ] && [ "$mem_percent" -lt 50 ]; then
            status="🟡 Partially used"
        else
            status="🔴 In use"
        fi
        
        printf "GPU %s: %-15s | Memory: %5.1f/%5.1f GB (%3d%% free) | GPU Util: %3s%% | Mem Util: %3s%% | %s\n" \
            "$index" "$name" "$mem_used_gb" "$mem_total_gb" "$mem_free_percent" "$gpu_util" "$mem_util" "$status"
    done
    
    echo ""
}

# If node number is provided, check only that node
if [ -n "$1" ]; then
    check_node_gpus $1
else
    # Default: check both node 1 and node 2
    check_node_gpus 1
    check_node_gpus 2
fi


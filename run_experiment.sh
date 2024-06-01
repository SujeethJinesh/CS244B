#!/bin/bash

show_help() {
    echo "Usage: $0 [-h] [-o output_file] [-d experiment_duration] [-k server_kill_timeout] [-r server_recovery_timeout]"
    echo
    echo "  -h                           Display this help message."
    echo "  -o output_file               Specify the output file to log the results."
    echo "  -d experiment_duration       Specify the experiment duration in seconds."
    echo "  -k server_kill_timeout       Specify the server kill timeout in seconds (default is 15 seconds)."
    echo "  -r server_recovery_timeout   Specify the server recovery timeout in seconds (default is 15 seconds)."
}

# Default values
experiment_duration=120
server_kill_timeout=15
server_recovery_timeout=15

# Parse command-line arguments
output_file=""
while getopts "ho:d:k:r:" opt; do
    case $opt in
        h)
            show_help
            exit 0
            ;;
        o)
            output_file=$OPTARG
            ;;
        d)
            experiment_duration=$OPTARG
            ;;
        k)
            server_kill_timeout=$OPTARG
            ;;
        r)
            server_recovery_timeout=$OPTARG
            ;;
        \?)
            show_help
            exit 1
            ;;
    esac
done

if [ -n "$output_file" ]; then
    : > "$output_file"
fi

experiments=("SYNC_CONTROL" "ASYNC_CONTROL")

run_experiment() {
    local exp=$1
    echo "Running experiment: $exp"
    timeout $experiment_duration python3.11 main.py --experiment "$exp" --epochs=1000 --server_kill_timeout=$server_kill_timeout --server_recovery_timeout=$server_recovery_timeout 2>&1
    if [ $? -eq 124 ]; then
        echo "Experiment $exp timed out after $experiment_duration seconds."
    fi
}

for exp in "${experiments[@]}"; do
    if [ -z "$output_file" ]; then
        run_experiment "$exp"
    else
        {
            run_experiment "$exp"
        } | tee -a "$output_file"
    fi
    echo "Cooldown for 10 seconds..."
    sleep 10
done


# "SYNC_CONTROL": run_sync,
#   "ASYNC_CONTROL": run_async,
#   "SYNC_CHECKPOINTING": run_sync_object_store_checkpointing,
#   "ASYNC_CHECKPOINTING": run_async_object_store_checkpointing,
#   "SYNC_CHAIN_REPLICATION": run_sync_chain_replication,
#   "ASYNC_CHAIN_REPLICATION": run_async_chain_replication,
#   "ASYNC_RELAXED_CONSISTENCY": run_async_relaxed_consistency,



# python3.11 main.py --experiment SYNC_CONTROL --epochs=1000 --server_kill_timeout=15 --server_recovery_timeout=15
# python3.11 main.py --experiment ASYNC_CONTROL --epochs=1000 --server_kill_timeout=15 --server_recovery_timeout=15
# python3.11 main.py --experiment SYNC_CHECKPOINTING --epochs=1000 --server_kill_timeout=15 --server_recovery_timeout=15
# python3.11 main.py --experiment ASYNC_CHECKPOINTING --epochs=1000 --server_kill_timeout=15 --server_recovery_timeout=15

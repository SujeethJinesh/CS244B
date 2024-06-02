#!/bin/bash

show_help() {
    echo "Usage: $0 [-h] [-o output_file] [-d experiment_duration] [-k server_kill_timeout] [-r server_recovery_timeout]"
    echo
    echo "  -h                           Display this help message."
    echo "  -o output_file               Specify the output file to log the results."
    echo "  -d experiment_duration       Specify the experiment duration in seconds."
    echo "  -k server_kill_timeout       Specify the server kill timeout in seconds (default is 15 seconds)."
    echo "  -r server_recovery_timeout   Specify the server recovery timeout in seconds (default is 15 seconds)."
    # echo "  --kill_times kill_times      Specify the number of times to kill the server. (default is 1 times)"
}

# Default values
experiment_duration=120
server_kill_timeout=15
server_recovery_timeout=15
output_file=""
# kill_times = 1

# Current experiment process ID
current_experiment_pid=""

# Trap SIGTERM signal and perform cleanup
trap 'cleanup' SIGTERM

cleanup() {
    echo "Caught SIGTERM signal! Terminating running experiment..."
    if [ -n "$current_experiment_pid" ]; then
        kill -9 "$current_experiment_pid"
        echo "Experiment process $current_experiment_pid terminated."
    fi
    exit 1
}

# Parse command-line arguments
while getopts "ho:d:k:r:" opt; do
    case $opt in
        h)
            show_help
            exit 0
            ;;
        o)
            output_file=$OPTARG
            if [ -z "$output_file" ]; then
                :
            elif [ ! -e "$output_file" ]; then
                touch "$output_file"
            fi
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
        # -)
            # case "${OPTARG}" in
            #     kill_times)
            #         kill_times="${!OPTIND}"
            #         OPTIND=$((OPTIND + 1))
            #         ;;
            #     *)
            #         echo "Invalid option: --$OPTARG" >&2
            #         exit 1
            #         ;;
            # esac
            # ;;
        \?)
            show_help
            exit 1
            ;;
    esac
done

experiments=(
    "SYNC_CHECKPOINTING"
    # "ASYNC_CHECKPOINTING"
    # "SYNC_CHAIN_REPLICATION"
    # "ASYNC_CHAIN_REPLICATION"
    # "ASYNC_RELAXED_CONSISTENCY"
)

run_experiment() {
    local exp=$1
    echo "Running experiment: $exp"
    timeout \
    $experiment_duration \
    python3.11 \
    main.py \
    --experiment "$exp" \
    --epochs=1000 \
    --server_kill_timeout=$server_kill_timeout \
    --server_recovery_timeout=$server_recovery_timeout \
    # --kill_times=$kill_times \
    --kill_times=2 \
    2>&1

    current_experiment_pid=$!
    wait $current_experiment_pid
    if [ $? -eq 124 ]; then
        echo "Experiment $exp timed out after $experiment_duration seconds."
    fi
    current_experiment_pid=""
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

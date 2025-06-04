#!/bin/bash

lsof_result=$(lsof -i)

kill_process_by_port() {
    local port=$1
    local pids=$(echo "$lsof_result" | grep ":$port" | awk '{print $2}' | sort -u)

    if [ -n "$pids" ]; then
        for pid in $pids; do
            echo "Killing process $pid using port $port"
            kill $pid
        done
    else
        echo "No process found using port $port"
    fi
}

kill_process_by_port 9001
kill_process_by_port 36657
kill_process_by_port 36658
kill_process_by_port 36659
kill_process_by_port 36660
kill_process_by_port 36661
kill_process_by_port 36662
kill_process_by_port 36663
kill_process_by_port 36664
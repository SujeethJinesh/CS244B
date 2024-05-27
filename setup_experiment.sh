#!/bin/bash

# Set up runtime environment
# bash create_venv.sh

# Set up zookeeper
bash zoozookeeper_setup.sh
./apache-zookeeper-3.8.4-bin/bin/zkServer.sh start

# Setup Prometheus
ray metrics launch-prometheus

curl -O https://dl.grafana.com/oss/release/grafana-11.0.0.darwin-amd64.tar.gz
tar -zxvf grafana-11.0.0.darwin-amd64.tar.gz

# Setup Grafana
cd grafana-v11.0.0
./bin/grafana server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web



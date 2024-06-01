#!/bin/bash

# Set up runtime environment
# bash create_venv.sh

# Set up zookeeper
bash zoozookeeper_setup.sh
./apache-zookeeper-3.8.4-bin/bin/zkServer.sh start

# # Setup Prometheus
curl -O https://github.com/prometheus/prometheus/releases/download/v2.52.0/prometheus-2.52.0.darwin-amd64.tar.gz
cd prometheus-*
./prometheus --config.file=../metrics/prometheus.yml

# # Setup Grafana
curl -O https://dl.grafana.com/oss/release/grafana-11.0.0.darwin-amd64.tar.gz
tar -zxvf grafana-11.0.0.darwin-amd64.tar.gz

cd grafana-v11.0.0
cp ../metrics/grafana.ini grafana.ini
./bin/grafana server --config grafana.ini web


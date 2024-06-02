# CS244B

# Getting Started

## Running Experiments (Quick Start)

In order to run an experiment, you need to run the following script. This script
will set up the runtime environment, zookeeper server, Prometheus, and Grafana.

```
setup_experiment.sh
```

This following script runs all experiments with 0, 1, and 2, parameter server failures with the given configuration.

```
run_experiment.sh -o [output_file] -d [experiment_duration] -k [server_kill_timeout] -r [server_recovery_timeout]
```

Note that if you want to kill the script and the processes that it spawns, you can first run

```
pskill -f run_experiment.sh 
```
and then send a SIGTERM signal to the bash process.

To run individual experiment, you can use the following command.

```
python3 main.py --experiment {SYNC_CONTROL,ASYNC_CONTROL,ASYNC_CHAIN_REPLICATION,ASYNC_RELAXED_CONSISTENCY,DEBUG_NO_CHECKPOINTING,DEBUG_DISK_CHECKPOINTING,DEBUG_OBJECT_STORE_CHECKPOINTING} [--model {IMAGENET,DEBUG}] [--kill_times 1]
```

Specify `--kill_times` to enumerate the number of times to kill the parameter server.

## Runtime Environment

To create the required Python virtual environment, run

```
./create_venv.sh
```

To activate the virtual environment, use the following command

```
source training_env/bin/activate
```

## Zookeeper

A zookeeper installation script is provided at `zookeeper/zookeeper_setup.sh`. After running this script, `apache-zookeeper-3.8.4-bin` should appear in the top-level of this project.

A running zookeeper server is needed before start the chain node experiments.

A local zookeeper server can be started by running:

```
./apache-zookeeper-3.8.4-bin/bin/zkServer.sh start
```

To stop the zookeeper server, run

```
./apache-zookeeper-3.8.4-bin/bin/zkServer.sh stop
```

To get debug information from the zookeeper server, you can also run the server
in the foreground.

```
./apache-zookeeper-3.8.4-bin/bin/zkServer.sh start-foreground
```

By default, the zookeeper clients used in the chain node experiments listen to
localhost at `127.0.0.1:2181`.

## Metrics

Steps for setting up metrics monitoring on a single-node local cluster:

1. Download [Prometheus](https://prometheus.io/download/). Unzip the
   archive into a local directory.

2. Download [Grafana](https://grafana.com/grafana/download). Unzip the
   archive into a local directory.

3. Start the ray cluster experiment, e.g. `python main.py`.

4. Open the ray dashboard in the browser. The default address is
   `localhost:8265`.

4.5 Verify the metrics are correct by verifying `localhost:8081` and grepping for `ray_training_loss` or `ray_training_accuracy`

5. In a new terminal, navigate to the Prometheus directory (e.g.
   `prometheus-2.52.0-rc.1.darwin-amd64`), run:

   ```
    ray metrics launch-prometheus
   ```

   This starts a service that records metrics from ray.

6. In another new terminal, navigate to the Grafana directory (e.g.
   `/grafana-v10.4.2/`), run:

   ```
   ./bin/grafana server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web`
   ```

   This starts the Grafana server to visualize the metrics with
   presets.

7. Open `localhost:3000` in the browser to access Grafana. The
   "Cluster Utilization" chart in the Serve Dashboard is often the
   most relevant.

8. Open `localhost:9090` in the browser to access Prometheus. It
   provides more granular dashboarding/metric monitoring than Grafana.
   Some useful queries include `ray_node_cpu_utilization`,
   `ray_node_mem_used`, `ray_component_uss_mb`. The full list of
   metrics is defined in [the official ray doc](https://docs.ray.io/en/latest/ray-observability/reference/system-metrics.html).

9. Check the node details in the Ray dashboard to see resource usage
   for individual workers.

10. (Optional) By default, metrics are collected every 10 seconds. To
    change this, edit `/tmp/ray/session_latest/metrics/prometheus/prometheus.yml`.
    Setting a long interval might make it harder to quickly notice
    issues like dead worker nodes.

## Create Grafana Dashboards

1. First login to grafana on the top right corner with admin/admin.

2. Create a dashboard with the `+` icon on the top right corner and scroll down to `Select Metric`.

3. Add the metric you prefer like `ray_training_...` and then set the experiment equal to the experiment name.

4. You can modify what the dashboard shows (timeseries graph, table, etc.). When exporting data, prefer using a table view.

5. (Optional) You can save this panel in your dashboard view.

## Custom Metrics

We instrument metrics for this project in addition to Ray's default metrics. We export metrics using Prometheus and provide a custom Grafana dashboard to visual our experiment results. To import the our pre-configured dashboard, you need to start the Prometheus server and then in the Grafana UI, you can create a new dashboard by importing `metrics/dashboard.json`.

To Verify that custom metrics are implemented correctly, you can open `localhost:8081` to see if there are logs associated with your custom metrics.

## GCP

You must first authenticate by installing `gcloud` using https://cloud.google.com/sdk/docs/install and running:

```
gcloud auth application-default login
```

Next, run the following:

```
cd /path/to/CS244B

gcloud config set project ft-distributed-ml-training

ray up gcp.yaml

# View the dashboard at http://127.0.0.1:8265/ after running:
ray dashboard /path/to/CS244B/gcp.yaml

# In a separate terminal, run:
ray job submit --runtime-env-json='{"excludes": ["/.git/"]}' --working-dir . -- python main.py
```

Useful commands:

```
  # To terminate the cluster:
    ray down gcp.yaml

  # To retrieve the IP address of the cluster head:
    ray get-head-ip gcp.yaml

  # To port-forward the cluster's Ray Dashboard to the local machine:
    ray dashboard gcp.yaml

  # To submit a job to the cluster, port-forward the Ray Dashboard in another terminal and run:
    ray job submit --address http://localhost:<dashboard-port> --working-dir . -- python my_script.py

  # To connect to a terminal on the cluster head for debugging:
    ray attach gcp.yaml

  # To monitor autoscaling:
    ray exec gcp.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```

### Set up GCP

SSH to cluster head `ray attach gcp.yaml` and run the following:

```
# Java
# https://cloud.google.com/java/docs/setup#debianubuntu

# ZooKeeper

pip install kazoo
wget "https://dlcdn.apache.org/zookeeper/zookeeper-3.8.4/apache-zookeeper-3.8.4-bin.tar.gz"
tar -xzf apache-zookeeper-3.8.4-bin.tar.gz
rm apache-zookeeper-3.8.4-bin.tar.gz
# By default zookeeper uses zoo.cfg as the server starting config.
mv apache-zookeeper-3.8.4-bin/conf/zoo_sample.cfg apache-zookeeper-3.8.4-bin/conf/zoo.cfg

./apache-zookeeper-3.8.4-bin/bin/zkServer.sh start

# Prometheus

wget https://github.com/prometheus/prometheus/releases/download/v2.52.0/prometheus-2.52.0.linux-amd64.tar.gz
tar -xzf prometheus-2.52.0.linux-amd64.tar.gz
rm prometheus-2.52.0.linux-amd64.tar.gz

# Grafana

wget https://dl.grafana.com/enterprise/release/grafana-enterprise-11.0.0.linux-amd64.tar.gz
tar -zxvf grafana-enterprise-11.0.0.linux-amd64.tar.gz

# [DO NOT RUN] TO FORCE KILL
ray stop --force
```

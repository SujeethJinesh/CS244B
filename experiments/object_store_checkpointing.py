import ray
import time

from metrics.metric_exporter import MetricExporter
from parameter_servers.model_saver import ModelSaver
from parameter_servers.server_actor import ParameterServer
from parameter_servers.server_killer import kill_server

LEARNING_RATE = 1e-2

def run_sync_object_store_checkpointing(model, num_workers, epochs, server_kill_timeout, server_recovery_timeout):
    ms = ModelSaver.remote()
    metric_exporter = MetricExporter.remote("sync checkpointing")
    
    def _run_experiment(first_run=True):
        try:
            ps = ParameterServer.remote(LEARNING_RATE, metric_exporter=metric_exporter, model_saver=ms)
            if not first_run:
                ray.get(ps.set_weights.remote(ms.get_weights.remote(), ms.get_iteration_count.remote()))
            else:
                server_killer_ref = kill_server.remote([ps], server_kill_timeout)
            ray.get(ps.run_synch_experiment.remote(num_workers))
        except Exception as e:
            print('catching exception', e)
            ps = None
            time.sleep(server_recovery_timeout)
            _run_experiment(first_run=False)
    
    _run_experiment()

def run_async_object_store_checkpointing(model, num_workers, epochs, server_kill_timeout, server_recovery_timeout):
    ms = ModelSaver.remote()
    metric_exporter = MetricExporter.remote("async checkpointing")
    
    def _run_experiment(first_run=True):
        try:
            ps = ParameterServer.remote(LEARNING_RATE, metric_exporter=metric_exporter, model_saver=ms)
            if not first_run:
                ray.get(ps.set_weights.remote(ms.get_weights.remote(), ms.get_iteration_count.remote()))
            else:
                server_killer_ref = kill_server.remote([ps], server_kill_timeout)
            ray.get(ps.run_asynch_experiment.remote(num_workers))
        except Exception as e:
            print('catching exception', e)
            ps = None
            time.sleep(server_recovery_timeout)
            _run_experiment(first_run=False)
    
    _run_experiment()
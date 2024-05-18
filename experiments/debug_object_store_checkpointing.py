import ray

from parameter_servers.model_saver import ModelSaver
from parameter_servers.server_actor import ParameterServer
from parameter_servers.server_killer import kill_server

LEARNING_RATE = 1e-2

def run_debug_object_store_checkpointing(model, ckpoint_period_sec: float = 10):
  ms = ModelSaver.remote()
  def _run_experiment(first_run=True):
    try:
      ps = ParameterServer.remote(LEARNING_RATE, model_saver=ms)
      if not first_run:
        ray.get(ps.set_weights.remote(ms.get_weights.remote(), ms.get_iteration_count.remote()))
      else:
        server_killer_ref = kill_server.remote([ps], 30)
      ray.get(ps.run_training.remote(False))
    except Exception as e:
      print('catching exception', e)
      ps = None
      _run_experiment(first_run=False)
  _run_experiment()
import ray

from parameter_servers.server_actor import ParameterServer

LEARNING_RATE = 1e-2

def run_debug_no_checkpointing(model):
  ps = ParameterServer.remote(LEARNING_RATE)
  ray.get([ps.run_training.remote(False)])

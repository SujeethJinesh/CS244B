import ray
import time
import threading

from experiments.synch import run_synch_experiment
from experiments.asynch import run_asynch_experiment
from parameter_servers.server_actor import ParameterServer, ModelSaver

@ray.remote
def kill_server(actor_handle_list, timeout_sec=10):
    time.sleep(timeout_sec)
    print("disrupt thread wakes up")
    for actor_handle in actor_handle_list:
    	ray.kill(actor_handle)
    print("killed server successfully")


def main():
  # Run asynchronous param server experiment
  ray.init(ignore_reinit_error=True)
  model_saver = ModelSaver.remote()
  ps1 = ParameterServer.remote(1e-2, 1, model_saver)
  ps2 = ParameterServer.remote(1e-2, 2, model_saver)
  # ray.get([ps.run_asynch_experiment.remote()])
  try:
    ray.get([ps1.run_synch_experiment.remote(), kill_server.remote([ps1], 10)])
  except Exception as e:
    print("Catching exception", e)
    ray.get([ps2.run_synch_experiment.remote()])
  # ray.get([ps.run_asynch_experiment_with_chain_replication.remote()])

if __name__ == "__main__":
  main()
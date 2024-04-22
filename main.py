import ray
import time

from experiments.synch import run_synch_experiment
from experiments.asynch import run_asynch_experiment
from parameter_servers.server_actor import ParameterServer

def main():
  # Run synch.py
  # run_synch_experiment()

  # Run asynch.py
  # run_asynch_experiment()

  # Run asynchronous param server experiment
  ray.init(ignore_reinit_error=True)
  ps = ParameterServer.remote(1e-2)
  ray.get([ps.run_synch_experiment.remote()])
  # ray.get([ps.run_asynch_experiment.remote()])

if __name__ == "__main__":
  main()
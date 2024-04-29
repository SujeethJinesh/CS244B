import ray
import time
import threading

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
  # ray.get([ps.run_asynch_experiment.remote()])
  ray.get([ps.run_synch_experiment.remote()])
  # ray.get([ps.run_asynch_experiment_with_chain_replication.remote()])

  # ps = ParameterServer.options(max_concurrency=2).remote(1e-2)
  # try: 
  #   ray.get([ps.run_synch_experiment.remote(), ps.exit.remote(10)])
  # except:
  #   print("An exception occured")

  # def disrupt_training(actor_handle):
  #   time.sleep(10)
  #   print("disrupt thread wakes up")
  #   ray.kill(actor_handle)

  # thread = threading.Thread(target=disrupt_training, args=(ps,))
  # thread.start()
  # thread.join()

  print("Driver exits")

if __name__ == "__main__":
  main()
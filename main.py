import ray
import time
import threading

from experiments.synch import run_synch_experiment
from experiments.asynch import run_asynch_experiment
from parameter_servers.server_actor import ParameterServer
from parameter_servers.server_killer import kill_server

LEARNING_RATE = 1e-2
SYNCHRONOUS = True

def run_experiment_with_no_ckpointing(ps: ParameterServer):
  ray.get([ps.run_training.remote(SYNCHRONOUS)])

def run_experiment_with_object_store_ckpointing(ps: ParameterServer, ckpoint_period_sec: float = 10):
  try:
    done_id, result_ids = [], [ps.run_training.remote(SYNCHRONOUS)]
    current_weights = ray.get([ps.get_weights.remote()])
    current_weights_ref = ray.put(*current_weights)
    print(current_weights)
    print(done_id)
    while len(done_id) < 1:
      print('saving weights')
      done_id, result_ids = ray.wait(result_ids, timeout=ckpoint_period_sec)
      print(done_id, result_ids)
      current_weights = ray.get([ps.get_weights.remote()])
      current_weights_ref = ray.put(*current_weights)
      print('saving weights')
      print(current_weights_ref)
  except:
    print('catching exception')
    # Initialize new ParameterServer.
    ps = ParameterServer.remote(LEARNING_RATE)
    print('making new parameter server')
    if current_weights_ref:
      # Recover from checkpoint.
      print('Recovering')
      ray.get([ps.set_weights.remote(current_weights_ref)])
      print('running again')
      run_experiment_with_object_store_ckpointing(ps)
    

def run_experiment_with_disk_ckpointing(ps: ParameterServer):
  print('not implemented')



def main():
  # Run synch.py
  # run_synch_experiment()

  # Run asynch.py
  # run_asynch_experiment()

  # Run asynchronous param server experiment
  ray.init(ignore_reinit_error=True)
  ps = ParameterServer.remote(LEARNING_RATE)
  # ray.get([ps.run_asynch_experiment.remote()])

  # server_killer_ref = kill_server.remote([ps], 30)

  run_experiment_with_object_store_ckpointing(ps)
  # ray.get(server_killer_ref)

  # ps = ParameterServer.options(max_concurrency=2).remote(1e-2)
  # try: 
  #   ray.get([ps.run_synch_experiment.remote(), ps.exit.remote(10)])
  # except:
  #   print("An exception occured")

  print("Driver exits")

if __name__ == "__main__":
  main()
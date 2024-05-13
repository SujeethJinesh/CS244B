import ray
import time
import threading
import os
import tempfile
import shutil


from experiments.synch import run_synch_experiment
from experiments.asynch import run_asynch_experiment
# from kazoo.client import KazooClient
# from kazoo.recipe.barrier import Barrier
from parameter_servers.server_actor import ParameterServer
from parameter_servers.server_actor_disk_ckpoint import ParameterServerDiskCkpoint
from parameter_servers.server_killer import kill_server
from parameter_servers.model_saver import ModelSaver

LEARNING_RATE = 1e-2
SYNCHRONOUS = False

def run_experiment_with_no_ckpointing():
  ps = ParameterServer.remote(LEARNING_RATE)
  ray.get([ps.run_training.remote(SYNCHRONOUS)])
    

def run_experiment_with_disk_ckpointing():
  checkpoint_dir = tempfile.mkdtemp()
  ps = ParameterServerDiskCkpoint.remote(LEARNING_RATE, checkpoint_dir)
  server_killer_ref = kill_server.remote([ps], 10, no_restart=False)
  ray.get(ps.run_training.remote(SYNCHRONOUS))
  shutil.rmtree(checkpoint_dir)


def run_experiment_with_object_store_ckpointing(ckpoint_period_sec: float = 10):
  ms = ModelSaver.remote()
  def _run_experiment(first_run=True):
    try:
      ps = ParameterServer.remote(LEARNING_RATE, model_saver=ms)
      if not first_run:
        ray.get(ps.set_weights.remote(ms.get_weights.remote(), ms.get_iteration_count.remote()))
      else:
        server_killer_ref = kill_server.remote([ps], 30)
      ray.get(ps.run_training.remote(SYNCHRONOUS))
    except Exception as e:
      print('catching exception', e)
      ps = None
      _run_experiment(first_run=False)
  _run_experiment()
    

def run_chain_node_experiment():
  ray.init(ignore_reinit_error=True)

  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()

  ps_dict = {}

  for i in range(1, 4):
      ps = ParameterServer.remote(1e-2, node_id=i)
      ps_dict[i] = ps
      # Ensures all zookeeper paths associated with the chain nodes exist.
      while not zk.exists("/base/" + str(i)):
        time.sleep(2)

  def run_new_primary():
    print("New primary runs")
    minimum = 100
    for node in zk.get_children('/base'):
      if int(node) < minimum:
        minimum = int(node)
    if minimum in ps_dict:
      primary = ps_dict[minimum]
      try:
        # ray.get([primary.run_synch_chain_node_experiment.remote(), kill_server.remote([primary], 10)])
        ray.get([primary.run_asynch_chain_node_experiment.remote(), kill_server.remote([primary], 10)])
      except Exception as e:
        print("Catching exception", e)
        # Ray and Zookeeper uses different communication channels, 
        # so synchronizatoin is needed here.
        print("block on node ", '/base/' + str(minimum))
        barrier = Barrier(client=zk, path='/base/' + str(minimum))
        barrier.wait()
        return False
    else:
      print("nothing gets run")
    return True

  time.sleep(5)

  while True:
    if run_new_primary():
      return

def main():
  # Run asynchronous param server experiment
  print('excluding')
  ray.init()
  
  # ray.get([ps.run_asynch_experiment.remote()])

  run_experiment_with_no_ckpointing()
  # ray.get(server_killer_ref)
  # run_experiment_with_object_store_ckpointing()
  # ray.get(server_killer_ref)
  # ray.get([ps.run_synch_experiment.remote()])
  # ray.get([ps.run_asynch_experiment_with_chain_replication.remote()])

  # ps = ParameterServer.options(max_concurrency=2).remote(1e-2)
  # try: 
  #   ray.get([ps.run_synch_experiment.remote(), ps.exit.remote(10)])
  # except:
  #   print("An exception occured")

  print("Driver exits")


if __name__ == "__main__":
  main()
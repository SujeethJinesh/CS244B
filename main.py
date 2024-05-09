import ray
import time
import threading

from experiments.synch import run_synch_experiment
from experiments.asynch import run_asynch_experiment
from parameter_servers.server_actor import ParameterServer
from kazoo.client import KazooClient
from kazoo.recipe.barrier import Barrier

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
  
  ps_dict = {}
  ps1 = ParameterServer.remote(1e-2, 1)
  ps_dict[1] = ps1
  ps2 = ParameterServer.remote(1e-2, 2)
  ps_dict[2] = ps2
  ps3 = ParameterServer.remote(1e-2, 3)
  ps_dict[3] = ps3

  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()

  def run_new_primary():
    print("New primary runs")
    minimum = 100
    for node in zk.get_children('/base'):
      if int(node) < minimum:
        minimum = int(node)
    if minimum in ps_dict:
      primary = ps_dict[minimum]
      try:
        ray.get([primary.run_synch_experiment.remote(), kill_server.remote([primary], 10)])
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

if __name__ == "__main__":
  main()
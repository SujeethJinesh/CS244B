import time

import ray
from kazoo.client import KazooClient
from kazoo.recipe.barrier import Barrier

from parameter_servers.model_saver import PartitionedStore
from parameter_servers.server_actor import ParameterServer
from parameter_servers.server_killer import kill_server

def run_async_chain_replication(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  num_chain_nodes = 3

  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()

  store = PartitionedStore.remote(num_chain_nodes)
  ps_dict = {}

  for i in range(num_chain_nodes):
      ps = ParameterServer.remote(1e-2, node_id=i, ref_store=store)
      ps_dict[i] = ps
      # Ensures all zookeeper paths associated with the chain nodes exist.
      while not zk.exists("/exp3/" + str(i)):
        time.sleep(2)

  def run_new_primary():
    print("New primary runs")
    minimum = 100
    for node in zk.get_children('/exp3'):
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
        print("block on node ", '/exp3/' + str(minimum))
        barrier = Barrier(client=zk, path='/exp3/' + str(minimum))
        barrier.wait()
        return False
    else:
      print("nothing gets run")
    return True

  time.sleep(5)

  while True:
    if run_new_primary():
      return
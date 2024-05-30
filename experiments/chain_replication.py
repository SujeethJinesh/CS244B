import threading
import time

import ray
from kazoo.client import KazooClient
from kazoo.recipe.barrier import Barrier

from parameter_servers.model_saver import PartitionedStore
from parameter_servers.server_actor import ParameterServer
from parameter_servers.server_killer import kill_server
from metrics.metric_exporter import MetricExporter

def run_chain_replication(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5, sync=False):
  num_chain_nodes = 3

  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()

  metric_exporter = MetricExporter.remote("chain replication")

  ps_dict = {}

  for i in range(num_chain_nodes):
    ps = ParameterServer.remote(1e-2, node_id=i, metric_exporter=metric_exporter)
    ps_dict[i] = ps
    # Ensures all zookeeper paths associated with the chain nodes exist.
    while True:
      if zk.exists("/exp3/" + str(i)):
        metric_exporter.set_zookeeper_reads.remote(1)  # Update read metric
        break
      time.sleep(2)

  def run_new_primary():
    print("New primary runs")
    minimum = 100
    children = zk.get_children('/exp3')
    metric_exporter.set_zookeeper_reads.remote(1)  # Update read metric
    for node in children:
      if int(node) < minimum:
        minimum = int(node)
    if minimum in ps_dict:
      primary = ps_dict[minimum]
      try:
        if sync:
          if minimum > 0 or server_kill_timeout <= 0:
            ray.get([primary.run_synch_chain_node_experiment.remote(num_workers)])
          else:
            ray.get([primary.run_synch_chain_node_experiment.remote(num_workers), kill_server.remote([primary], server_kill_timeout)])
        else:
          if minimum > 0 or server_kill_timeout <= 0:
             ray.get([primary.run_asynch_chain_node_experiment.remote(num_workers)])
          else: 
            ray.get([primary.run_asynch_chain_node_experiment.remote(num_workers), kill_server.remote([primary], server_kill_timeout)])
         
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

  def recover_server(timeout, node_id, metric_exporter, ps_dict):
      time.sleep(timeout)
      ps = ParameterServer.remote(1e-2, node_id=node_id, metric_exporter=metric_exporter)
      ps_dict[node_id] = ps

  for i in range(num_chain_nodes - 1):
    run_new_primary()
    
    threading.Thread(
        target=recover_server, 
        args=(server_recovery_timeout, num_chain_nodes + i, metric_exporter, ps_dict)
    ).start()

  run_new_primary()

def run_async_chain_replication(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  run_chain_replication(model, num_workers, epochs, server_kill_timeout, server_kill_timeout, False)

def run_sync_chain_replication(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  run_chain_replication(model, num_workers, epochs, server_kill_timeout, server_kill_timeout, True)

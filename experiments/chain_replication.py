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
          ray.get([primary.run_synch_chain_node_experiment.remote(num_workers, metric_exporter), kill_server.remote([primary], server_kill_timeout)])
        else:
          # ray.get([primary.run_asynch_chain_node_experiment.remote(num_workers), kill_server.remote([primary], server_kill_timeout)])
          ray.get([primary.run_asynch_chain_node_experiment.remote(num_workers, metric_exporter)])
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

  for i in range(num_chain_nodes - 1):
    run_new_primary()
    ps = ParameterServer.remote(1e-2, node_id=num_chain_nodes + i, metric_exporter=metric_exporter)
    time.sleep(server_recovery_timeout)
  run_new_primary()

def run_async_chain_replication(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  run_chain_replication(model, num_workers, epochs, server_kill_timeout, server_kill_timeout, False)

def run_sync_chain_replication(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  run_chain_replication(model, num_workers, epochs, server_kill_timeout, server_kill_timeout, True)

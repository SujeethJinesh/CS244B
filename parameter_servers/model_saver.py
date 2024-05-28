import ray
import time

from collections import defaultdict
from kazoo.client import KazooClient
from queue import Queue
from parameter_servers.server_actor import ParameterServer

MAX_QUEUE_SIZE = 50

# Used as a global actor in tandem with Ray's object store to store remote
# objects' references for a long enough time so that they don't get garbage
# collected.
@ray.remote
class PartitionedStore(object):
    def __init__(self, partition_size, partition_array = None):
        self.partitioned_store = {}
        self.partition_zk_clients = {}
        
        if partition_array is None:
            partition_array = list(range(partition_size))
        for k in partition_array:
            self.partitioned_store[k] = Queue(maxsize=MAX_QUEUE_SIZE)
            # self.partition_zk_clients[partition] = KazooClient(hosts='127.0.0.1:2181')
            # self.partition_zk_clients[partition].start()
            # self.partition_zk_clients[partition].exists(partition, watch=self.store_data_ref)

    def partitioned_store_put(self, partition, value):
        self.partitioned_store[partition].put(value)
        print("after put partitioned store has size ", self.partitioned_store[partition].qsize(), " for ", partition)
    # def store_data_ref(event):
    #     print("Store data ref")
    #     if event.type == "CHANGED":
    #         retrieved_data = self.partition_zk_clients[event.path].get(event.path, watch=self.store_data_ref)
    #         object_ref = ray.cloudpickle.loads(retrieved_data[0])
    #         partitioned_store_put(event.path, object_ref)
    


@ray.remote
class ModelSaver(object):
    def __init__(self):
        self.weight_reference = None
        self.iteration_count = 0
        self.old_weight_id = None

    def set_weights_iteration_count(self, weights, iteration_count):
        print('saving weights')
        self.weight_reference = ray.put(weights)
        self.iteration_count = iteration_count

    def set_weights(self, w):
      self.weight_reference = ray.put(w)
      pickled_weight_id = ray.cloudpickle.dumps(self.weight_reference)
      if self.old_weight_id:
        ray.internal.free([self.old_weight_id])
        del self.old_weight_id
        self.old_weight_id = pickled_weight_id
      return pickled_weight_id

    def get_weights(self):
        return ray.get(self.weight_reference)

    def get_iteration_count(self):
        return self.iteration_count

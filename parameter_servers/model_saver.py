import ray
import time

from parameter_servers.server_actor import ParameterServer

@ray.remote
class ModelSaver(object):
    def __init__(self):
        self.weight_reference = None
        self.iteration_count = 0

    def set_weights_iteration_count(self, weights, iteration_count):
        print('saving weights')
        self.weight_reference = ray.put(weights)
        self.iteration_count = iteration_count

    def set_weights(self, w):
      self.weight_reference = ray.put(w)
      pickled_weight_id = ray.cloudpickle.dumps(self.weight_reference)
      return pickled_weight_id

    def get_weights(self):
        return ray.get(self.weight_reference)

    def get_iteration_count(self):
        return self.iteration_count
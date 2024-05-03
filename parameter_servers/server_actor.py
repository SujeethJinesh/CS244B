import torch
import numpy as np
from models.test_model import ConvNet
import ray
import time
import pickle
from workers.worker_task import compute_gradients
from models.test_model import ConvNet, get_data_loader, evaluate
from zookeeper.zoo import KazooChainNode

iterations = 200
num_workers = 2
weight_update_frequency = 10

@ray.remote
class ModelSaver(object):
    def __init__(self):
        self.weight_reference = None

    def set_weights(self, weights):
        print('model saver saving weights')
        self.weight_reference = ray.put(weights)
        return self.weight_reference

    def get_weights(self):
        print("enter get weights")
        if self.weight_reference is not None:
            print("weight reference is ", self.weight_reference)
            print("type is ", type(self.weight_reference))
            return ray.get(self.weight_reference)

@ray.remote
class ParameterServer(object):
    def __init__(self, lr, node_id, model_saver):
        self.model_saver = model_saver
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.chain_node = KazooChainNode(node_id, [], self.retrieve_weights_from_zookeeper)
        time.sleep(2)

    def apply_gradients(self, gradients):
        grad = ray.get(gradients)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*grad)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()
    
    # TODO potentially race condition: retrieval weights not being triggered
    def store_weights_in_zookeeper(self, weights):
      print("start storing weights")
      id_w = ray.put(weights)
      #id_w = self.model_saver.set_weights.remote(weights)
      pickled_weight_id = ray.cloudpickle.dumps(id_w)
      print(self.chain_node.node_id)
      self.chain_node.zk.set("/base/" + str(self.chain_node.node_id), pickled_weight_id)
         
      # TODO: store the value in zookeeper
      # zookeeper.put(pickled_weight_id)

    def retrieve_weights_from_zookeeper(self, event):
      # TODO: implement the following function
      # zid = get_ray_weight_id(event)
      node_id = event.path[6]
      if event.type=='CHANGED' and node_id < self.chain_node.node_id:
        retrieved_data = self.chain_node.zk.get("/base/" + node_id)
        unpickled_id_w_string = ray.cloudpickle.loads(retrieved_data[0])
        new_weights = ray.get(unpickled_id_w_string)
        # new_weights = self.model_saver.get_weights.remote()
        self.model.set_weights(new_weights)
        print("backup recieve weights")
        self.chain_node.zk.exists("/base/"+str(node_id), watch=self.chain_node.handle_delete_or_change_event)

    def run_synch_experiment(self):
      test_loader = get_data_loader()[1]

      print("Running synchronous parameter server training.")
      current_weights = self.get_weights()
      for i in range(iterations):
          gradients = [compute_gradients.remote(current_weights) for _ in range(num_workers)]
          # Calculate update after all gradients are available.
          current_weights = self.apply_gradients(gradients)
          
          if i % weight_update_frequency == 0:
              self.store_weights_in_zookeeper(current_weights)
              # Evaluate the current model.
              self.model.set_weights(current_weights)
              accuracy = evaluate(self.model, test_loader)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

      print("Final accuracy is {:.1f}.".format(accuracy))

    def run_asynch_experiment(self):
      test_loader = get_data_loader()[1]

      print("Running Asynchronous Parameter Server Training.")
      current_weights = self.get_weights()
      gradients = []
      for _ in range(num_workers):
          gradients.append(compute_gradients.remote(current_weights))

      for i in range(iterations * num_workers):
          ready_gradient_list, _ = ray.wait(gradients)
          ready_gradient_id = ready_gradient_list[0]
          gradients.remove(ready_gradient_id)

          # Compute and apply gradients.
          current_weights = self.apply_gradients([ready_gradient_id])
          gradients.append(compute_gradients.remote(current_weights))

          if i % weight_update_frequency == 0:
              # Evaluate the current model after every 10 updates.
              self.model.set_weights(current_weights)
              accuracy = evaluate(self.model, test_loader)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

      print("Final accuracy is {:.1f}.".format(accuracy))

    def run_asynch_experiment_with_chain_replication(self):
      test_loader = get_data_loader()[1]

      print("Running Asynchronous Parameter Server Training.")
      current_weights = self.get_weights()
      gradients = []
      for _ in range(num_workers):
          gradients.append(compute_gradients.remote(current_weights))

      for i in range(iterations * num_workers):
          ready_gradient_list, _ = ray.wait(gradients)
          ready_gradient_id = ready_gradient_list[0]
          gradients.remove(ready_gradient_id)

          # Compute and apply gradients.
          current_weights = self.apply_gradients([ready_gradient_id])
          gradients.append(compute_gradients.remote(current_weights))

          if i % weight_update_frequency == 0:
              # Evaluate the current model after every 10 updates.
              self.model.set_weights(current_weights)
              accuracy = evaluate(self.model, test_loader)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

      print("Final accuracy is {:.1f}.".format(accuracy))

    def sleep(self, sleep_sec):
        # for i in range(10):
        #     time.sleep(sleep_sec)
        while True:
            pass

    def exit(self, sleep_sec):
        print("in exit method")
        time.sleep(sleep_sec)
        ray.actor.exit_actor()

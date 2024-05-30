import torch
import numpy as np
from models.test_model import ConvNet
import ray
import time
import os
from workers.worker_task import compute_gradients
# from models.test_model import ConvNet, get_data_loader, evaluate
from models.fashion_mnist import FashionMNISTConvNet, fashion_mnist_get_data_loader
from models.model_common import evaluate
# from models.cifar10 import ResNet, get_data_loader, evaluate
from zookeeper.zoo import KazooChainNode

# TODO (Change to training epochs)
iterations = 4000
# Set weight update frequency for the model maintained by the parameter server.
# Used for model evaluation.
WEIGHT_UPDATE_FREQUENCY = 10

@ray.remote(max_restarts=0)
class ParameterServer(object):
    def __init__(self, lr, node_id=None, metric_exporter=None, model_saver=None):
        self.model = FashionMNISTConvNet()
        self.start_iteration = 0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.start_iteration = 0
        if model_saver is not None:
          self.model_saver = model_saver
        if metric_exporter is not None:
          self.metric_exporter = metric_exporter
        if node_id is not None:
          self.chain_node = KazooChainNode(node_id, [], self.retrieve_weights_from_zookeeper)

    def apply_gradients(self, gradients):
        grad = ray.get(gradients)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*grad)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        if self.metric_exporter is not None:
            self.metric_exporter.set_gradients_processed.remote(len(grad))
        return self.model.get_weights()

    def set_weights(self, weights, iteration_count):
        if weights:
          self.model.set_weights(weights)
          self.start_iteration = iteration_count

    def get_weights(self):
        return self.model.get_weights()
        
    def store_weights_in_zookeeper(self, weights, iteration):
      print("Node " + str(self.chain_node.node_id) + " starts storing weights")
      id_w = ray.put([weights, iteration + 1])
      pickled_weight_id = ray.cloudpickle.dumps(id_w)
      self.chain_node.zk.set("/exp3/" + str(self.chain_node.node_id), pickled_weight_id)
      if self.metric_exporter is not None:
          self.metric_exporter.set_zookeeper_writes.remote(1)  # Update write metric

    def store_weights_in_zookeeper_chain(self, weights_ref):
      print("Node " + str(self.chain_node.node_id) + " starts storing weights")
      self.chain_node.zk.set("/exp3/" + str(self.chain_node.node_id), weights_ref)
      if self.metric_exporter is not None:
          self.metric_exporter.set_zookeeper_writes.remote(1)  # Update write metric

    def retrieve_weights_from_zookeeper(self, event):
      node_id = event.path[6]
      if event.type == 'CHANGED' and int(node_id) < self.chain_node.node_id:
        retrieved_data = self.chain_node.zk.get("/exp3/" + node_id)
        if self.metric_exporter is not None:
            self.metric_exporter.set_zookeeper_reads.remote(1)  # Update read metric
        unpickled_id_w_string = ray.cloudpickle.loads(retrieved_data[0])
        new_weights, iteration = ray.get(unpickled_id_w_string)
        self.model.set_weights(new_weights)
        self.start_iteration = iteration
        self.store_weights_in_zookeeper_chain(retrieved_data[0])
        print("backup receive weights")
        self.chain_node.zk.exists("/exp3/" + str(node_id), watch=self.chain_node.handle_delete_or_change_event)
        if self.metric_exporter is not None:
            self.metric_exporter.set_zookeeper_reads.remote(1)  # Update read metric

    def run_synch_chain_node_experiment(self, num_workers):
      test_loader = fashion_mnist_get_data_loader()[1]

      print("Running synchronous parameter server training.")
      current_weights = self.get_weights()
      for i in range(self.start_iteration, iterations):
          gradients = [compute_gradients.remote(current_weights, self.metric_exporter) for _ in range(num_workers)]
          # Calculate update after all gradients are available.
          current_weights = self.apply_gradients(gradients, metric_exporter=self.metric_exporter)
          
          if i % WEIGHT_UPDATE_FREQUENCY == 0:
              self.store_weights_in_zookeeper(current_weights, i)
              # Evaluate the current model.
              self.set_weights(current_weights, i)
              accuracy = evaluate(self.model, test_loader)
              self.metric_exporter.set_accuracy.remote(accuracy)
              print("Iter {}: \taccuracy is {:.3f}".format(i, accuracy))

      print("Final accuracy is {:.3f}.".format(accuracy))

    def run_asynch_chain_node_experiment(self, num_workers):
      test_loader = fashion_mnist_get_data_loader()[1]

      print("Running Asynchronous Parameter Server Training.")
      current_weights = self.get_weights()
      gradients = []
      for _ in range(num_workers):
          gradients.append(compute_gradients.remote(current_weights, self.metric_exporter))
      for i in range(self.start_iteration, iterations * num_workers):
          ready_gradient_list, _ = ray.wait(gradients)
          ready_gradient_id = ready_gradient_list[0]
          gradients.remove(ready_gradient_id)

          # Compute and apply gradients.
          current_weights = self.apply_gradients([ready_gradient_id], metric_exporter=self.metric_exporter)
          gradients.append(compute_gradients.remote(current_weights))

          if i % WEIGHT_UPDATE_FREQUENCY == 0:
              # Evaluate the current model after every 10 updates.
              self.store_weights_in_zookeeper(current_weights, i)
              self.set_weights(current_weights, i)
              accuracy = evaluate(self.model, test_loader)
              self.metric_exporter.set_accuracy.remote(accuracy)
              print("Iter {}: \taccuracy is {:.3f}".format(i, accuracy))

      print("Final accuracy is {:.1f}.".format(accuracy))
    
    def run_synch_experiment(self, num_workers):
      test_loader = fashion_mnist_get_data_loader()[1]

      print("Running synchronous parameter server training.")
      current_weights = self.get_weights()
      for i in range(self.start_iteration, iterations):
          gradients = [compute_gradients.remote(current_weights, self.metric_exporter) for _ in range(num_workers)]
          # Calculate update after all gradients are available.
          current_weights = self.apply_gradients(gradients)
          
          if i % WEIGHT_UPDATE_FREQUENCY == 0:
              # Evaluate the current model.
              self.set_weights(current_weights, i)
              accuracy = evaluate(self.model, test_loader)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

          if i % 100 == 0:
            # save checkpoint
            self.model_saver.set_weights_iteration_count.remote(current_weights, i)


      print("Final accuracy is {:.1f}.".format(accuracy))

    def run_asynch_experiment(self, num_workers):
      test_loader = fashion_mnist_get_data_loader()[1]

      print("Running Asynchronous Parameter Server Training.")
      current_weights = self.get_weights()
      gradients = []
      for _ in range(num_workers):
          gradients.append(compute_gradients.remote(current_weights, self.metric_exporter))
      for i in range(self.start_iteration, iterations * num_workers):
          ready_gradient_list, _ = ray.wait(gradients)
          ready_gradient_id = ready_gradient_list[0]
          gradients.remove(ready_gradient_id)

          # Compute and apply gradients.
          current_weights = self.apply_gradients([ready_gradient_id])
          gradients.append(compute_gradients.remote(current_weights))

          if i % WEIGHT_UPDATE_FREQUENCY == 0:
              # Evaluate the current model after every 10 updates.
              self.set_weights(current_weights, i)
              accuracy = evaluate(self.model, test_loader)
              self.metric_exporter.set_accuracy.remote(accuracy)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
          
          if i % 100 == 0:
            # save checkpoint
            self.model_saver.set_weights_iteration_count.remote(current_weights, i)
            


      print("Final accuracy is {:.1f}.".format(accuracy))

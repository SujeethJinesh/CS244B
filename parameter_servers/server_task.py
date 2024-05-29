import time
import numpy as np
import torch
import ray
# from models.test_model import get_data_loader, evaluate
from models.fashion_mnist import fashion_mnist_get_data_loader
from models.model_common import evaluate
from kazoo.client import KazooClient
from kazoo.exceptions import NodeExistsError, NoNodeError

iterations = 400
weight_update_frequency = 10

@ray.remote
class ParamServerTaskActor:
  
  def __init__(self):
    pass

  def _start_zk(self, metric_exporter):
    zk = KazooClient(hosts='127.0.0.1:2181', timeout=1.0)
    zk.start()
    zk.create("/base/parameter_server", b"", ephemeral=True, makepath=True)
    if metric_exporter is not None:
      metric_exporter.set_zookeeper_writes.remote(1)  # Update write metric
    return zk

  def _load_weights_for_optimizer(self, zk, model, lr, metric_exporter):
    retrieved_data = zk.get("/base/weights")
    metric_exporter.set_zookeeper_reads.remote(1)  # Update read metric
    unpickled_w_string = ray.cloudpickle.loads(retrieved_data[0])
    stored_weights = ray.get(unpickled_w_string)
    model.set_weights(stored_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer

  def run_parameter_server_task(self, model, num_workers, lr, weight_saver, metric_exporter):
    print("Parameter Server is starting")
    then = time.time()
    test_loader = fashion_mnist_get_data_loader()[1]

    zk = self._start_zk(metric_exporter)
    model, optimizer = self._load_weights_for_optimizer(zk, model, lr, metric_exporter)

    def maybe_retrieve_gradients_from_zk(event):
      nonlocal zk

      # If there's no longer a lock on the gradients path
      if event.type=='DELETED':
        try:
          worker_index = event.path.split("/")[-1]
          worker_grad_path = f"/base/gradients/{worker_index}"

          # Lock the gradients so we can read them
          zk.create(event.path, ephemeral=True, makepath=True)

          # Read the current list of gradient updates in the zookeeper node
          remote_grad_updates = None
          pickled_remote_gradient_updates_id = zk.get(worker_grad_path)
          metric_exporter.set_zookeeper_reads.remote(1)  # Update read metric
          if pickled_remote_gradient_updates_id[0] != b'':
            remote_grad_updates_ref = ray.cloudpickle.loads(pickled_remote_gradient_updates_id[0])
            remote_grad_updates = ray.get(remote_grad_updates_ref)
            ray.internal.free([remote_grad_updates_ref])
            del remote_grad_updates_ref

          # Place gradients in object store
          id_grads = ray.put(b'')
          pickled_grad_id = ray.cloudpickle.dumps(id_grads)
          zk.set(worker_grad_path, pickled_grad_id)
          metric_exporter.set_zookeeper_writes.remote(1)  # Update write metric

          # Unlock the gradients
          zk.delete_async(event.path)
          metric_exporter.set_zookeeper_writes.remote(1)  # Update write metric
          return remote_grad_updates
        except NodeExistsError:
          # We'll get em next time
          print(f"PS Lock on node {worker_index} was locked, we'll get em next time.")
        except NoNodeError:
          print(f"No node found, pass")
      return None

    def apply_gradients(grad):
      nonlocal model, optimizer
      # print(f"Applying gradients of length {len(grad)}")
      if grad:
        temp_optimizer = optimizer
        if len(grad) > 10:
          temp_optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*grad)
        ]
        temp_optimizer.zero_grad()
        model.set_gradients(summed_gradients)
        temp_optimizer.step()
        metric_exporter.set_gradients_processed.remote(len(grad))
        return model.get_weights()
      return None

    def store_weights_in_zookeeper(w):
      nonlocal model, zk, weight_saver
      model.set_weights(w)
      pickled_id_w = ray.get(weight_saver.set_weights.remote(w))

      zk.set("/base/weights", pickled_id_w)
      metric_exporter.set_zookeeper_writes.remote(1)  # Update write metric

    def evaluate_model():
      nonlocal then, model, test_loader
      accuracy = evaluate(model, test_loader)
      print("accuracy is {:.1f}".format(accuracy))
      metric_exporter.set_accuracy.remote(accuracy)

    def handle_gradient_update(event):
      nonlocal then
      weights = None
      gradients = maybe_retrieve_gradients_from_zk(event)
      if gradients:
        weights = apply_gradients(gradients)
      if weights:
        store_weights_in_zookeeper(weights)
        now = time.time()
        if now - then > 1.0:
          evaluate_model()
          then = now

      zk.exists(event.path, watch=handle_gradient_update)
      metric_exporter.set_zookeeper_reads.remote(1)  # Update write metric

    print("Running initial evaluation")
    evaluate_model()

    for worker_index in range(num_workers):
      zk.exists(f"/base/gradients/lock/{worker_index}", watch=handle_gradient_update)
      metric_exporter.set_zookeeper_reads.remote(1)  # Update write metric

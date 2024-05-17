import time
import numpy as np
import torch
from models.test_model import ConvNet
import ray
import os
import threading
from models.test_model import ConvNet, get_data_loader, evaluate
from kazoo.client import KazooClient
from kazoo.exceptions import NodeExistsError, NoNodeError

iterations = 400
weight_update_frequency = 10

@ray.remote
class ParamServerTaskActor:
  
  def __init__(self):
    pass

  def run_parameter_server_task(self, model, num_workers, lr, weight_saver):
    print("Parameter Server is starting")
    then = time.time()
    test_loader = get_data_loader()[1]

    zk = KazooClient(hosts='127.0.0.1:2181', timeout=1.0)
    zk.start()
    zk.create("/base/parameter_server", b"", ephemeral=True, makepath=True)

    retrieved_data = zk.get("/base/weights")
    unpickled_w_string = ray.cloudpickle.loads(retrieved_data[0])
    stored_weights = ray.get(unpickled_w_string)
    model.set_weights(stored_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def maybe_retrieve_gradients_from_zk(event):
      nonlocal zk

      # If there's no longer a lock on the 
      if event.type=='DELETED':
        try:
          worker_index = event.path.split("/")[-1]
          worker_grad_path = f"/base/gradients/{worker_index}"

          # Lock the gradients so we can read them
          zk.create(event.path, ephemeral=True, makepath=True)

          # Read the current list of gradient updates in the zookeeper node
          remote_grad_updates = None
          pickled_remote_gradient_updates_id = zk.get(worker_grad_path)
          if pickled_remote_gradient_updates_id[0] != b'':
            remote_grad_updates_ref = ray.cloudpickle.loads(pickled_remote_gradient_updates_id[0])
            remote_grad_updates = ray.get(remote_grad_updates_ref)

          # Place gradients in object store
          id_grads = ray.put(b'')
          pickled_grad_id = ray.cloudpickle.dumps(id_grads)
          zk.set(worker_grad_path, pickled_grad_id)

          # Unlock the gradients
          zk.delete(event.path)
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
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*grad)
        ]
        optimizer.zero_grad()
        model.set_gradients(summed_gradients)
        optimizer.step()
        return model.get_weights()
      return None

    def store_weights_in_zookeeper(w):
      nonlocal model, zk, weight_saver
      model.set_weights(w)
      pickled_id_w = ray.get(weight_saver.set_weights.remote(w))

      zk.set("/base/weights", pickled_id_w)

    def evaluate_model():
      nonlocal then, model, test_loader
      accuracy = evaluate(model, test_loader)
      print("accuracy is {:.1f}".format(accuracy))

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

    print("Running initial evaluation")
    evaluate_model()

    for worker_index in range(num_workers):
      zk.exists(f"/base/gradients/lock/{worker_index}", watch=handle_gradient_update)

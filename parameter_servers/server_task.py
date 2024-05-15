import time
import numpy as np
import torch
from models.test_model import ConvNet
import ray
import os
import threading
from models.test_model import ConvNet, get_data_loader, evaluate
from kazoo.client import KazooClient

iterations = 400
weight_update_frequency = 10

@ray.remote(max_retries=0)
def run_parameter_server_task(model, num_workers, lr):
  print("Parameter Server is starting")
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)

  then = time.time()
  test_loader = get_data_loader()[1]

  zk = KazooClient(hosts='127.0.0.1:2181', timeout=1.0)
  zk.start()
  zk.create("/base/parameter_server", b"", ephemeral=True, makepath=True)

  def retrieve_gradients_from_zk(event):
    nonlocal zk
    # print(f"Got gradient notification from event: {event}")
    if event.type=='CHANGED':
      retrieved_data = zk.get(event.path)
      unpickled_grad_string = ray.cloudpickle.loads(retrieved_data[0])
      # TODO need to properly pass back gradient updates for array of updates.
      grads = ray.get(unpickled_grad_string)
      return [grads]

  def apply_gradients(grad):
    nonlocal model, optimizer
    # print(f"Applying gradient to weight")
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
    nonlocal model, zk
    # print("PS storing weights in zookeeper")
    model.set_weights(w)
    id_w = ray.put(w)
    pickled_weight_id = ray.cloudpickle.dumps(id_w)

    # This will likely need to be done in an actor so that the weights data will still be kept
    zk.set("/base/weights", pickled_weight_id)

  def evaluate_model():
    nonlocal then, model, test_loader
    now = time.time()
    if now - then > 2.0:
      # Evaluate the current model after every 10 seconds.
      accuracy = evaluate(model, test_loader)
      print("accuracy is {:.1f}".format(accuracy))
      then = now

  def handle_gradient_update(event):
    gradients = retrieve_gradients_from_zk(event)
    weights = apply_gradients(gradients)
    if weights:
      store_weights_in_zookeeper(weights)
      evaluate_model()

    print(f"pid is {os.getpid()} and thread is {threading.get_ident()} and ray task id is {ray.get_runtime_context().get_task_id()}")
    zk.exists(event.path, watch=handle_gradient_update)

  for worker_index in range(num_workers):
    zk.exists(f"/base/gradients/{worker_index}", watch=handle_gradient_update)

  return os.getpid(), ray.get_runtime_context().get_task_id()

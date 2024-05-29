import ray.cloudpickle
import torch.nn.functional as F
import torch.nn as nn
from models.fashion_mnist import FashionMNISTConvNet, fashion_mnist_get_data_loader
from models.test_model import TestModel, test_model_get_data_loader
from kazoo.client import KazooClient
from kazoo.exceptions import NodeExistsError
import ray
import json
from threading import Thread

@ray.remote
def compute_gradients(model_name, weights, metric_exporter=None):
    if model_name == "FASHION":
      model = FashionMNISTConvNet()
      data_loader_fn = fashion_mnist_get_data_loader
    else:
      model = TestModel()
      data_loader_fn = test_model_get_data_loader
    data_iterator = iter(data_loader_fn()[0])

    model.train()
    model.set_weights(weights)
    try:
        data, target = next(data_iterator)
    except StopIteration:  # When the epoch ends, start a new epoch.
        data_iterator = iter(data_loader_fn()[0])
        data, target = next(data_iterator)
    model.zero_grad()
    output = model(data)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    if metric_exporter is not None:
      metric_exporter.set_loss.remote(loss.item())
    loss.backward()
    return model.get_gradients()

@ray.remote
def compute_gradients_relaxed_consistency(model_name, model, worker_index, epochs=5, metric_exporter=None):
  #TODO add data loader func
  if model_name == "FASHION":
    data_loader_fn = fashion_mnist_get_data_loader
  else:
    data_loader_fn = test_model_get_data_loader
  data_iterator = iter(data_loader_fn()[0])
  curr_epoch = 0
  print(f"Worker {worker_index} is starting at Epoch {curr_epoch}")
  data_iterator = iter(data_loader_fn()[0])
  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()

  local_gradient_updates = []

  worker_grad_path = f"/base/gradients/{worker_index}"
  worker_grad_lock_path = f"/base/gradients/lock/{worker_index}"
  zk.create(worker_grad_path, b"", ephemeral=True, makepath=True)

  def get_weights():
    nonlocal zk
    retrieved_data = zk.get("/base/weights")
    unpickled_w_string = ray.cloudpickle.loads(retrieved_data[0])
    return ray.get(unpickled_w_string)

  def put_gradients():
    nonlocal zk, worker_grad_path, local_gradient_updates

    try:
      # Lock the gradient update node
      zk.create(worker_grad_lock_path, ephemeral=True, makepath=True)

      # Read the current list of gradient updates in the zookeeper node
      remote_grad_updates = []
      pickled_remote_gradient_updates_id = zk.get(worker_grad_path)
      if pickled_remote_gradient_updates_id[0] != b'':
        remote_grad_updates_ref = ray.cloudpickle.loads(pickled_remote_gradient_updates_id[0])
        remote_grad_updates = ray.get(remote_grad_updates_ref)
        if remote_grad_updates == b'':
          remote_grad_updates = []
        ray.internal.free([remote_grad_updates_ref])
        del remote_grad_updates_ref
      remote_grad_updates.extend(local_gradient_updates)

      # Place gradients in object store
      id_grads = ray.put(remote_grad_updates)
      pickled_grad_id = ray.cloudpickle.dumps(id_grads)
      zk.set(worker_grad_path, pickled_grad_id)

      # Unlock the gradient update node and reset local gradient updates
      zk.delete_async(worker_grad_lock_path)
      local_gradient_updates = []
    except NodeExistsError:
      # We'll get em next time
      print(f"Lock on node {worker_index} was locked, we'll get em next time.")

  def compute_grads(data, target):
    nonlocal data_iterator, model
    try:
        data, target = next(data_iterator)
    except StopIteration:  # When the epoch ends, start a new epoch.
        data_iterator = iter(data_loader_fn()[0])
        data, target = next(data_iterator)
    model.zero_grad()
    output = model(data)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    if metric_exporter is not None:
      metric_exporter.set_loss.remote(loss.item())
    loss.backward()
    return model.get_gradients()

  def has_next_data():
    nonlocal curr_epoch, epochs, data_iterator
    try:
      d, t = next(data_iterator)
      return True, d, t
    except StopIteration:
      if curr_epoch < epochs:
        data_iterator = iter(data_loader_fn()[0])
        d, t = next(data_iterator)
        curr_epoch += 1
        print(f"Starting Epoch {curr_epoch}")
        return True, d, t
      else:
        return False, None, None

  while True:
    has, data, target = has_next_data()
    if not has:
      break
    weights = get_weights()
    model.set_weights(weights)
    gradients = compute_grads(data, target)
    local_gradient_updates.append(gradients)

    # Parameter server is done doing work on the gradients
    if not zk.exists(worker_grad_lock_path):
      put_gradients()

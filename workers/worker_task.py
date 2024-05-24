import torch
import ray.cloudpickle
import torch.nn.functional as F
import torch.nn as nn
from models.fashion_mnist import ConvNet, get_data_loader
# from models.test_model import ConvNet, get_data_loader
from kazoo.client import KazooClient
from kazoo.exceptions import NodeExistsError
import ray
import json
from threading import Thread

@ray.remote
def compute_gradients(weights):
    model = ConvNet()
    data_iterator = iter(get_data_loader()[0])

    model.train()
    model.set_weights(weights)
    try:
        data, target = next(data_iterator)
    except StopIteration:  # When the epoch ends, start a new epoch.
        data_iterator = iter(get_data_loader()[0])
        data, target = next(data_iterator)
    model.zero_grad()
    output = model(data)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    print("loss is", loss)
    loss.backward()
    return model.get_gradients()

@ray.remote
def compute_gradients_relaxed_consistency(model, worker_index, epochs=5):
  curr_epoch = 0
  print(f"Worker {worker_index} is starting at Epoch {curr_epoch}")
  data_iterator = iter(get_data_loader()[0])
  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()

  if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
      print("MPS not available because the current PyTorch install was not "
            "built with MPS enabled.")
    else:
      print("MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine.")
  else:
    mps_device = torch.device("mps")
    x = torch.ones(5, device=mps_device)
    y = x * 2
    model.to(mps_device)

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
        data_iterator = iter(get_data_loader()[0])
        data, target = next(data_iterator)
    model.zero_grad()
    output = model(data)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    loss.backward()
    return model.get_gradients()

  def has_next_data():
    nonlocal curr_epoch, epochs, data_iterator
    try:
      d, t = next(data_iterator)
      return True, d, t
    except StopIteration:
      if curr_epoch < epochs:
        data_iterator = iter(get_data_loader()[0])
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

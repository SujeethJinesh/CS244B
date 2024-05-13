import torch.nn.functional as F
from models.test_model import ConvNet, get_data_loader
from kazoo.client import KazooClient
import ray

@ray.remote
def compute_gradients(weights):
    model = ConvNet()
    data_iterator = iter(get_data_loader()[0])

    model.set_weights(weights)
    try:
        data, target = next(data_iterator)
    except StopIteration:  # When the epoch ends, start a new epoch.
        data_iterator = iter(get_data_loader()[0])
        data, target = next(data_iterator)
    model.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return model.get_gradients()

@ray.remote
def compute_gradients_relaxed_consistency(worker_index):
  model = ConvNet()
  data_iterator = iter(get_data_loader()[0])

  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()
  zk.create(f"/base/gradients/{worker_index}", b"", ephemeral=True, makepath=True)

  def get_weights():
    print(f"Getting weights")
    retrieved_data = zk.get("/base/weights")
    unpickled_w_string = ray.cloudpickle.loads(retrieved_data[0])
    return ray.get(unpickled_w_string)

  def put_gradients(grads):
    print("PS storing grads in zookeeper")
    id_grad = ray.put(grads)
    pickled_grad_id = ray.cloudpickle.dumps(id_grad)
    zk.set(f"/base/gradients/{worker_index}", pickled_grad_id)

  def compute_grads():
    global data_iterator
    try:
        data, target = next(data_iterator)
    except StopIteration:  # When the epoch ends, start a new epoch.
        data_iterator = iter(get_data_loader()[0])
        data, target = next(data_iterator)
    model.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return model.get_gradients()

  while True:
    model.set_weights(get_weights())
    gradients = compute_grads()
    put_gradients(gradients)

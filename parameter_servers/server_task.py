import datetime
import numpy as np
from models.test_model import ConvNet
import ray
from models.test_model import ConvNet, get_data_loader, evaluate
from kazoo.client import KazooClient

iterations = 400
weight_update_frequency = 10

@ray.remote
def run_parameter_server_task(num_workers):
  model = ConvNet()
  then = datetime.datetime.now()
  test_loader = get_data_loader()[1]

  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()
  zk.create("/base/parameter_server", b"", ephemeral=True, makepath=True)

  def retrieve_gradients_from_zk(event):
    print(f"Got gradient notification from {event.path}")
    if event.type=='CHANGED':
      retrieved_data = zk.get(event.path)
      unpickled_grad_string = ray.cloudpickle.loads(retrieved_data[0])
      # TODO need to properly pass back gradient updates for array of updates.
      grads = ray.get(unpickled_grad_string)
      print("Getting gradients from worker")
      print(grads)
      return [grads]

  def apply_gradients(grad):
    print(f"Applying gradient to weight")
    summed_gradients = [
        np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*grad)
    ]
    model.optimizer.zero_grad()
    model.set_gradients(summed_gradients)
    model.optimizer.step()
    return model.get_weights()

  def store_weights_in_zookeeper(w):
    print("PS storing weights in zookeeper")
    model.set_weights(w)
    id_w = ray.put(w)
    pickled_weight_id = ray.cloudpickle.dumps(id_w)
    zk.set("/base/weights", pickled_weight_id)

  def evaluate():
    if then - datetime.datetime.now() > datetime.timedelta(seconds = 10):
      # Evaluate the current model after every 10 seconds.
      accuracy = evaluate(model, test_loader)
      print(": \taccuracy is {:.1f}".format(accuracy))
      then = datetime.datetime.now()

  def handle_gradient_update(event):
    gradients = retrieve_gradients_from_zk(event)
    weights = apply_gradients(gradients)
    store_weights_in_zookeeper(weights)
    evaluate()

  for worker_index in range(num_workers):
    zk.exists(f"/base/gradients/{worker_index}", watch=handle_gradient_update)

  while True:
    pass

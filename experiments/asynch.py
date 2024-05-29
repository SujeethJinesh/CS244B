import ray
from parameter_servers.server_actor import ParameterServer
from workers.worker_task import compute_gradients
from metrics.metric_exporter import MetricExporter
# from models.test_model import get_data_loader, evaluate
from models.fashion_mnist import fashion_mnist_get_data_loader
from models.model_common import evaluate

iterations = 200
num_workers = 2

def run_async(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  metric_exporter = MetricExporter.remote("async control")
  ps = ParameterServer.remote(1e-2)

  test_loader = fashion_mnist_get_data_loader()[1]

  print("Running Asynchronous Parameter Server Training.")
  current_weights = ps.get_weights.remote()
  gradients = []
  for _ in range(num_workers):
    gradients.append(compute_gradients.remote(current_weights, metric_exporter))

  for i in range(iterations * num_workers * epochs):
    ready_gradient_list, _ = ray.wait(gradients)
    ready_gradient_id = ready_gradient_list[0]
    gradients.remove(ready_gradient_id)

    # Compute and apply gradients.
    current_weights = ps.apply_gradients.remote([ready_gradient_id], metric_exporter)
    gradients.append(compute_gradients.remote(current_weights, metric_exporter))

    if i % 10 == 0:
      # Evaluate the current model after every 10 updates.
      model.set_weights(ray.get(current_weights))
      accuracy = evaluate(model, test_loader)
      print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
      metric_exporter.set_accuracy.remote(accuracy)

  print("Final accuracy is {:.1f}.".format(accuracy))
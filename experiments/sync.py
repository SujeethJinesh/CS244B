import ray
from parameter_servers.server_actor import ParameterServer
from workers.worker_task import compute_gradients
from metrics.metric_exporter import MetricExporter
# from models.test_model import get_data_loader, evaluate
from models.fashion_mnist import fashion_mnist_get_data_loader
from models.model_common import evaluate

iterations = 200
num_workers = 2

def run_sync(model_name, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  metric_exporter = MetricExporter.remote("sync control")
  ps = ParameterServer.remote(model_name, 1e-2)
  data_loader_fn = fashion_mnist_get_data_loader
  # TODO Update data_loader_fn
  test_loader = fashion_mnist_get_data_loader[1]

  print("Running synchronous parameter server training.")
  current_weights = ps.get_weights.remote()
  for i in range(iterations * epochs):
    gradients = [compute_gradients.remote(current_weights, metric_exporter=metric_exporter) for _ in range(num_workers)]
    # Calculate update after all gradients are available.
    current_weights = ps.apply_gradients.remote(gradients)

    if i % 10 == 0:
        # Evaluate the current model.
        model.set_weights(ray.get(current_weights))
        accuracy = evaluate(model, test_loader)
        print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
        metric_exporter.set_accuracy.remote(accuracy)

  print("Final accuracy is {:.1f}.".format(accuracy))

  # Clean up Ray resources and processes before the next example.
  ray.shutdown()
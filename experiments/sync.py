import ray
from parameter_servers.server_actor import ParameterServer
from workers.worker_task import compute_gradients
from metrics.metric_exporter import MetricExporter
import copy
import threading
from evaluation.evaluator import async_eval
from evaluation.evaluator_state import evaluator_state
from shared import MODEL_MAP, evaluate
from models.fashion_mnist import FashionMNISTConvNet, fashion_mnist_get_data_loader

iterations = 200
num_workers = 2

def run_sync(model_name, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  metric_exporter = MetricExporter.remote("sync control")
  if model_name == "FASHION_MNIST":
    model = FashionMNISTConvNet()
    _, test_loader = fashion_mnist_get_data_loader()
  else:
    model = None
    test_loader = None
  ps = ParameterServer.remote(model_name, 1e-2)

  # Start eval thread
  timer_runs = threading.Event()
  timer_runs.set()
  eval_thread = threading.Thread(target=async_eval, args=(timer_runs, model_name, test_loader, metric_exporter, evaluate))
  eval_thread.start()

  print("Running synchronous parameter server training.")
  weights_ref = ps.get_weights.remote()
  for _ in range(iterations * epochs):
    gradients = [compute_gradients.remote(model_name, weights_ref, metric_exporter=metric_exporter) for _ in range(num_workers)]
    # Calculate update after all gradients are available.
    weights_ref = ps.apply_gradients.remote(gradients)

    evaluator_state.weights_lock.acquire()
    evaluator_state.CURRENT_WEIGHTS = ray.get(weights_ref)
    model.set_weights(evaluator_state.CURRENT_WEIGHTS)
    evaluator_state.weights_lock.release()

  timer_runs.clear()
  eval_thread.join()  # Ensure the eval thread has finished

  # Clean up Ray resources and processes before the next example.
  ray.shutdown()

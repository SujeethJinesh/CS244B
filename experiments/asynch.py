import ray
from parameter_servers.server_actor import ParameterServer
from workers.worker_task import compute_gradients
from metrics.metric_exporter import MetricExporter
from models.fashion_mnist import get_data_loader, evaluate
import threading
import copy
from evaluation.evaluator import async_eval
from evaluation.evaluator_state import evaluator_state

iterations = 200
num_workers = 2

def run_async(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5):
  metric_exporter = MetricExporter.remote("async control")
  ps = ParameterServer.remote(1e-2)

  test_loader = get_data_loader()[1]

  # Start eval thread
  model_copy = copy.deepcopy(model)
  timer_runs = threading.Event()
  timer_runs.set()
  eval_thread = threading.Thread(target=async_eval, args=(timer_runs, model_copy, test_loader, metric_exporter, evaluate))
  eval_thread.start()

  print("Running Asynchronous Parameter Server Training.")
  current_weights = ps.get_weights.remote()
  gradients = []
  for _ in range(num_workers):
    gradients.append(compute_gradients.remote(current_weights))

  for _ in range(iterations * num_workers * epochs):
    ready_gradient_list, _ = ray.wait(gradients)
    ready_gradient_id = ready_gradient_list[0]
    gradients.remove(ready_gradient_id)

    # Compute and apply gradients.
    current_weights = ps.apply_gradients.remote([ready_gradient_id])
    gradients.append(compute_gradients.remote(current_weights, metric_exporter=metric_exporter))

    evaluator_state.weights_lock.acquire()
    evaluator_state.CURRENT_WEIGHTS = ray.get(current_weights)
    evaluator_state.weights_lock.release()

  timer_runs.clear()
  eval_thread.join()  # Ensure the eval thread has finished

  # Clean up Ray resources and processes before the next example.
  ray.shutdown()

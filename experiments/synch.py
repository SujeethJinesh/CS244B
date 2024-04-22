import ray
from parameter_servers.server_actor import ParameterServer
from workers.worker_task import compute_gradients
from models.test_model import ConvNet, get_data_loader, evaluate

iterations = 200
num_workers = 2

def run_synch_experiment():
  ray.init(ignore_reinit_error=True)
  ps = ParameterServer.remote(1e-2)

  model = ConvNet()
  test_loader = get_data_loader()[1]

  print("Running synchronous parameter server training.")
  current_weights = ps.get_weights.remote()
  for i in range(iterations):
      gradients = [compute_gradients.remote(current_weights) for _ in range(num_workers)]
      # Calculate update after all gradients are available.
      current_weights = ps.apply_gradients.remote(*gradients)

      if i % 10 == 0:
          # Evaluate the current model.
          model.set_weights(ray.get(current_weights))
          accuracy = evaluate(model, test_loader)
          print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

  print("Final accuracy is {:.1f}.".format(accuracy))
  # Clean up Ray resources and processes before the next example.
  ray.shutdown()

import ray
from parameter_servers.server_actor import ParameterServer
from workers.worker_task import compute_gradients
from models.test_model import ConvNet, get_data_loader, evaluate

iterations = 200
num_workers = 2

def run_asynch_experiment():
  ray.init(ignore_reinit_error=True)
  ps = ParameterServer.remote(1e-2)

  model = ConvNet()
  test_loader = get_data_loader()[1]

  print("Running Asynchronous Parameter Server Training.")
  current_weights = ps.get_weights.remote()
  gradients = []
  for _ in range(num_workers):
      gradients.append(compute_gradients.remote(current_weights))

  for i in range(iterations * num_workers):
      ready_gradient_list, _ = ray.wait(gradients)
      ready_gradient_id = ready_gradient_list[0]
      gradients.remove(ready_gradient_id)

      # Compute and apply gradients.
      current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
      gradients.append(compute_gradients.remote(current_weights))

      if i % 10 == 0:
          # Evaluate the current model after every 10 updates.
          model.set_weights(ray.get(current_weights))
          accuracy = evaluate(model, test_loader)
          print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

  print("Final accuracy is {:.1f}.".format(accuracy))

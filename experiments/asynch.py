import ray
from parameter_servers.synchronous import ParameterServer
from workers.traditional import DataWorker
from models.test_model import ConvNet, get_data_loader, evaluate

iterations = 20_000
num_workers = 2

def run_asynch_experiment():
  ray.init(ignore_reinit_error=True)
  ps = ParameterServer.remote(1e-2)
  workers = [DataWorker.remote() for i in range(num_workers)]

  model = ConvNet()
  test_loader = get_data_loader()[1]

  print("Running Asynchronous Parameter Server Training.")
  current_weights = ps.get_weights.remote()
  gradients = {}
  for worker in workers:
      gradients[worker.compute_gradients.remote(current_weights)] = worker

  for i in range(iterations * num_workers):
      ready_gradient_list, _ = ray.wait(list(gradients))
      ready_gradient_id = ready_gradient_list[0]
      worker = gradients.pop(ready_gradient_id)

      # Compute and apply gradients.
      current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
      gradients[worker.compute_gradients.remote(current_weights)] = worker

      if i % 10 == 0:
          # Evaluate the current model after every 10 updates.
          model.set_weights(ray.get(current_weights))
          accuracy = evaluate(model, test_loader)
          print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

  print("Final accuracy is {:.1f}.".format(accuracy))

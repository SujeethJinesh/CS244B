import torch
import numpy as np
from models.test_model import ConvNet
import ray
import time
import os
from workers.worker_task import compute_gradients
from models.test_model import ConvNet, get_data_loader, evaluate

iterations = 200
num_workers = 2

@ray.remote(max_restarts=0)
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, gradients):
        grad = ray.get(gradients)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*grad)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_ckpoint(self):
      print('not implemented')
      # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
      #     checkpoint = None

      #     # Only the global rank 0 worker saves and reports the checkpoint
      #     if train.get_context().get_world_rank() == 0:
      #         ...  # Save checkpoint to temp_checkpoint_dir

      #         checkpoint = Checkpoint.from_directory(tmpdir)

      #     train.report(metrics, checkpoint=checkpoint)

    def run_training(self, synchronous=True):
      if synchronous:
        self.run_synch_training()
      else:
        self.run_asynch_training()


    def run_synch_training(self):
      test_loader = get_data_loader()[1]

      print("Running synchronous parameter server training.")
      current_weights = self.get_weights()
      for i in range(iterations):
          gradients = [compute_gradients.remote(current_weights) for _ in range(num_workers)]
          # Calculate update after all gradients are available.
          current_weights = self.apply_gradients(gradients)

          if i % 10 == 0:
              # Evaluate the current model.
              self.model.set_weights(current_weights)
              accuracy = evaluate(self.model, test_loader)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

          if i == 50:
            os._exit(0)

      print("Final accuracy is {:.1f}.".format(accuracy))

    def run_asynch_training(self):
      test_loader = get_data_loader()[1]

      print("Running Asynchronous Parameter Server Training.")
      current_weights = self.get_weights()
      gradients = []
      for _ in range(num_workers):
          gradients.append(compute_gradients.remote(current_weights))

      for i in range(iterations * num_workers):
          ready_gradient_list, _ = ray.wait(gradients)
          ready_gradient_id = ready_gradient_list[0]
          gradients.remove(ready_gradient_id)

          # Compute and apply gradients.
          current_weights = self.apply_gradients([ready_gradient_id])
          gradients.append(compute_gradients.remote(current_weights))

          if i % 10 == 0:
              # Evaluate the current model after every 10 updates.
              self.model.set_weights(current_weights)
              accuracy = evaluate(self.model, test_loader)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

      print("Final accuracy is {:.1f}.".format(accuracy))

    def exit(self, sleep_sec):
        print("in exit method")
        time.sleep(sleep_sec)
        os._exit(0)

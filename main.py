import argparse
import os
import random

import numpy as np
import torch
import ray

from experiments.async_relaxed_consistency import run_async_relaxed_consistency
from experiments.asynch import run_async
from experiments.sync import run_sync
from experiments.async_relaxed_consistency import run_async_relaxed_consistency

from experiments.chain_replication import run_async_chain_replication, run_sync_chain_replication
from experiments.debug_no_checkpointing import run_debug_no_checkpointing
from experiments.debug_disk_checkpointing import run_debug_disk_checkpointing
from experiments.debug_object_store_checkpointing import run_debug_object_store_checkpointing
# from models.test_model import ConvNet
from models.fashion_mnist import ConvNet

num_workers = 2

EXPERIMENT_MAP = {
  "SYNC_CONTROL": run_sync,
  "ASYNC_CONTROL": run_async,
  "SYNC_CHAIN_REPLICATION": run_sync_chain_replication,
  "ASYNC_CHAIN_REPLICATION": run_async_chain_replication,
  "ASYNC_RELAXED_CONSISTENCY": run_async_relaxed_consistency,

  "DEBUG_NO_CHECKPOINTING": run_debug_no_checkpointing,
  "DEBUG_DISK_CHECKPOINTING": run_debug_disk_checkpointing,
  "DEBUG_OBJECT_STORE_CHECKPOINTING": run_debug_object_store_checkpointing,
}

MODEL_MAP = {
  "IMAGENET": None,
  "DEBUG": ConvNet()
}

# TODO: This doesn't seem to make the randomness consistent
# We need to do more to ensure consistency.
def init_random_seeds(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


def main():
  # Initialize Ray as this is common between all experiments.
  print("Initializing Ray")
  runtime_env = {"pip": ["kazoo"]}
  print(ray.init(ignore_reinit_error=True, _metrics_export_port=8081, runtime_env=runtime_env))

  # Ensure consistency across experiments when it comes to randomness
  init_random_seeds()

  # Use flags for argument parsing
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment', type=str, choices=EXPERIMENT_MAP.keys(), required=True, help="Type of experiment to run")
  parser.add_argument('--model', type=str, choices=MODEL_MAP.keys(), default="debug", help="Type of model to use")
  parser.add_argument('--workers', type=int, default=1, help="Number of workers to use")
  parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to run")
  parser.add_argument('--server_kill_timeout', type=int, default=10, help="Time before parameter server is killed")
  parser.add_argument('--server_recovery_timeout', type=int, default=5, help="Time after parameter server is killed to recover")
  parser.add_argument('--use-mps', action='store_true', help="Use MPS for GPU acceleration on Apple Silicon")
  args = parser.parse_args()
  
  # Parse the flags
  experiment_name = args.experiment.upper()
  model_name = args.model.upper()
  has_mps = torch.backends.mps.is_available()
  use_mps_if_available = args.use_mps

  print(f"Does this device have MPS? {has_mps}")
  print(f"Using MPS? {use_mps_if_available}")

  device = torch.device("mps" if use_mps_if_available and has_mps else "cpu")

  experiment = EXPERIMENT_MAP[experiment_name]
  model = MODEL_MAP[model_name].to(device)
  workers = args.workers
  epochs = args.epochs
  server_kill_timeout = args.server_kill_timeout
  server_recovery_timeout = args.server_recovery_timeout

  # Run appropriate experiment
  print(f"Starting {experiment_name} experiment with model {model_name}.")
  experiment(model, num_workers=workers, epochs=epochs, server_kill_timeout=server_kill_timeout, server_recovery_timeout=server_recovery_timeout, device=device)
  print(f"Completed {experiment_name} experiment.")


if __name__ == "__main__":
  main()

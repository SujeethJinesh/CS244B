import subprocess
import time

import ray
from kazoo.client import KazooClient

from parameter_servers.model_saver import ModelSaver
from parameter_servers.server_killer import kill_server
from parameter_servers.server_task import ParamServerTaskActor
from workers.worker_task import compute_gradients_relaxed_consistency

WEIGHTS_ZK_PATH = "/base/weights"

def try_delete_zookeeper_weights_node():
  try:
    print(f"Attempting to delete {WEIGHTS_ZK_PATH} node if it exists")
    subprocess.run(
      ['./apache-zookeeper-3.8.4-bin/bin/zkCli.sh', 'delete', WEIGHTS_ZK_PATH],
      check=False,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True
    )
  except Exception as e:
    pass

def initialize_zk_with_weights(model):
  zk = KazooClient(hosts='127.0.0.1:2181', timeout=1.0)
  zk.start()

  weight_ref = ray.put(model.get_weights())
  weight_ref_string = ray.cloudpickle.dumps(weight_ref)

  try_delete_zookeeper_weights_node()
  zk.create(WEIGHTS_ZK_PATH, weight_ref_string, ephemeral=False, makepath=True)

def run_async_relaxed_consistency(model, num_workers=1):
  initialize_zk_with_weights(model)

  training_tasks = []

  # 0. Create WeightSaver
  weight_saver_ref = ModelSaver.remote()

  # 1. Create parameter server.
  ps_actor_ref = ParamServerTaskActor.remote()
  ps_ref = ps_actor_ref.run_parameter_server_task.remote(model, num_workers, 1e-3, weight_saver_ref)
  ray.get([ps_ref])

  # 2. Create workers.
  workers = [compute_gradients_relaxed_consistency.remote(model, i) for i in range(num_workers)]
  training_tasks.extend(workers)

  # 3. Kill Server.
  server_killer_ref = kill_server.remote([ps_actor_ref], timeout_sec=10, no_restart=True)
  ray.get([server_killer_ref])

  # 4. Recreate Server.
  time.sleep(5)
  recreated_ps_actor = ParamServerTaskActor.remote()
  recreated_ps_ref = recreated_ps_actor.run_parameter_server_task.remote(model, num_workers, 1e-3, weight_saver_ref)
  training_tasks.append(recreated_ps_ref)

  # 5. Run till completion
  ray.get(training_tasks)

  # 6. Cleanup
  try_delete_zookeeper_weights_node()
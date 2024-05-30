import shutil
import tempfile

import ray

from parameter_servers.server_actor_disk_ckpoint import ParameterServerDiskCkpoint
from parameter_servers.server_killer import kill_server

LEARNING_RATE = 1e-2

def run_debug_disk_checkpointing(model, num_workers=1, epochs=5, server_kill_timeout=10, server_recovery_timeout=5, ckpoint_period_sec: float = 10):
  checkpoint_dir = tempfile.mkdtemp()
  ps = ParameterServerDiskCkpoint.remote(LEARNING_RATE, checkpoint_dir)
  server_killer_ref = kill_server.remote([ps], 10, no_restart=False)
  ray.get(ps.run_training.remote(False))
  shutil.rmtree(checkpoint_dir)

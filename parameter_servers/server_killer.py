import ray
import time
import os
import signal

@ray.remote
def kill_server(handle_list, timeout_sec=10, no_restart=True):
  time.sleep(timeout_sec)
  print("disrupt thread wakes up")
  for handle in handle_list:
    ray.kill(handle, no_restart=no_restart)
  print("killed server successfully")

import ray
import time
import os
import signal

@ray.remote
def kill_server(handle_list, timeout_sec=10, no_restart=True, is_task=False, os_pid=None):
  time.sleep(timeout_sec)
  print("disrupt thread wakes up")
  for handle in handle_list:
    if is_task:
      # TODO: For some reason this ray.cancel doesn't actually kill the ps task
      print (f"killing handle {handle}")
      ray.cancel(handle, force=True, recursive=True)
      ray.cancel(handle, force=True, recursive=False)
      ray.cancel(handle, force=False, recursive=False)
      ray.cancel(handle, force=False, recursive=True)
      if os_pid:
        print(f"Killing pid {os_pid}")
        os.kill(os_pid, signal.SIGKILL)
      print ("killing task")
    else:
      ray.kill(handle, no_restart=no_restart)
  print("killed server successfully")

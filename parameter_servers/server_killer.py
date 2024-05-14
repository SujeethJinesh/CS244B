import ray
import time

@ray.remote
def kill_server(handle_list, timeout_sec=10, no_restart=True, is_task=False):
  time.sleep(timeout_sec)
  print("disrupt thread wakes up")
  for handle in handle_list:
    if is_task:
      # TODO: For some reason this ray.cancel doesn't actually kill the ps task
      print (f"killing handle {handle}")
      ray.cancel(handle, force=False, recursive=False)
      print ("killing task")
    else:
      ray.kill(handle, no_restart=no_restart)
  print("killed server successfully")

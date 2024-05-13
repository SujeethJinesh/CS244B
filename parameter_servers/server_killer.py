import ray
import time

@ray.remote
def kill_server(actor_handle_list, timeout_sec=10, no_restart=True):
    time.sleep(timeout_sec)
    print("disrupt thread wakes up")
    for actor_handle in actor_handle_list:
    	ray.kill(actor_handle, no_restart=no_restart)
    print("killed server successfully")
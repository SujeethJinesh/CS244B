import threading
import time
from zoo import KazooChainNode

def run_node(node_id, init_role):
    node = KazooChainNode(node_id, init_role)
    time.sleep(10)
    node.stop()

def fail_node(node_id, init_role):
    node = KazooChainNode(node_id, init_role)
    time.sleep(5)
    node.stop()

def check_state_change(node_id, init_role):
    node = KazooChainNode(node_id, init_role)
    print("NODE ", node_id, "has role: ", node.get_role())
    print("NODE ", node_id, "has prev node id: ", node.get_prev_id())
    print("NODE ", node_id, "has next node id: ", node.get_next_id())
    time.sleep(10)
    print("NODE ", node_id, "has role: ", node.get_role())
    print("NODE ", node_id, "has prev node id: ", node.get_prev_id())
    print("NODE ", node_id, "has next node id: ", node.get_next_id())
    node.stop()

def test_bring_up_chain():
    # Case 1  Successfully bring up a chain with prev and next set (test 3)
    print("Test Case: bringing up chain of 3")
    threads = []
    for i in range(1, 4):
        thread = threading.Thread(target=run_node, args=(i, []))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Successfully bring up a chain.")

def test_head_node_failure():
    # Case 2 Head node failure (frontend PS)
    print("Test Case: head node failure")
    threads = []
    thread1 = threading.Thread(target=fail_node, args=(1, ["head"]))
    thread2 = threading.Thread(target=check_state_change, args=(2, []))
    thread3 = threading.Thread(target=run_node, args=(3, ["tail"]))
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

def test_tail_node_failure():
    # Case 3 Tail node failure
    print("Test Case: tail node failure")
    threads = []
    thread1 = threading.Thread(target=run_node, args=(1, ["head"]))
    thread2 = threading.Thread(target=check_state_change, args=(2, []))
    thread3 = threading.Thread(target=fail_node, args=(3, ["tail"]))
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

def test_middle_node_failure():
    # Case 4 Middle node failure
    print("Test Case: middle node failure")
    threads = []
    thread1 = threading.Thread(target=check_state_change, args=(1, ["head"]))
    thread2 = threading.Thread(target=fail_node, args=(2, []))
    thread3 = threading.Thread(target=check_state_change, args=(3, ["tail"]))
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    test_bring_up_chain()
    test_head_node_failure()
    test_tail_node_failure()
    test_middle_node_failure()
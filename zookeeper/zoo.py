from kazoo.client import KazooClient

# Cases to handle for PS clients
# 1. Successfully bring up a chain with prev and next set (test 3)
# 2. Head node failure (frontend PS)
# 3. Tail node failure
# 4. Middle node failure

# 5. Recovery with head node failure
# 6. Recovery with tail node failure
# 7. Recovery after middle node failure


# TODO: What to do for Training Worker clients?

class KazooChainNode(object):
    def __init__(self, node_id, init_role, hosts='127.0.0.1:2181'):
        self.zk = KazooClient(hosts=hosts)
        self.zk.start()
        self.head = False
        self.tail = False
        self.prev_id = -1
        self.node_id = -1
        self.next_id = -1
        self.setup_node(node_id, init_role)

    def handle_event(self, event):
        print("event is", event)
        if event.type=='DELETED':
            # get node id
            if int(event.path[6]) < self.node_id:
                largest_smaller = None
                for node in self.zk.get_children("/base"):
                    if int(node) < self.node_id:
                        if largest_smaller is None or int(node) > largest_smaller:
                            largest_smaller = int(node)
                if largest_smaller is None:
                    self.set_head()
                else:
                    self.prev_id = largest_smaller
                    self.zk.exists("/base/"+str(self.prev_id), watch=self.handle_event)
            else:
                smallest_larger = None
                for node in self.zk.get_children("/base"):
                    if int(node) > self.node_id:
                        if smallest_larger is None or int(node) < smallest_larger:
                            smallest_larger = int(node)   
                if smallest_larger is None:
                    self.set_tail()
                else: 
                    self.next_id = smallest_larger
                    self.zk.exists("/base/"+str(self.next_id), watch=self.handle_event)

    def setup_node(self, node_id, init_role):
        print("NODE ", node_id, " is created")
        
        self.prev_id = node_id - 1
        self.node_id = node_id
        self.next_id = node_id + 1

        if "head" in init_role:
            self.set_head()
        if "tail" in init_role:
            self.set_tail()
        self.zk.create("/base/"+str(self.node_id), b"somevalue", ephemeral=True, makepath=True)
        self.zk.exists("/base/"+str(self.prev_id), watch=self.handle_event)
        self.zk.exists("/base/"+str(self.next_id), watch=self.handle_event)

    def stop(self):
        self.zk.stop()
    
    def set_head(self):
        self.head = True
        self.prev_id = -1

    def set_tail(self):
        self.tail = True
        self.next_id = -1

    def get_role(self):
        role = ""
        if not self.head and not self.tail:
            role = "middle"
        if self.head:
            role = "head, " + role
            role = role.strip(", ")
        if self.tail:
            role = role  + ", tail"
            role = role.strip(", ")
        return role
    
    def get_prev_id(self):
        return self.prev_id

    def get_next_id(self):
        return self.next_id





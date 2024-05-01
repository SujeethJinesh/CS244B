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
    def __init__(self, node_id, init_role, prev_node_change_callback, hosts='127.0.0.1:2181'):
        self.zk = KazooClient(hosts=hosts)
        self.zk.start()
        self.head = False
        self.tail = False
        self.prev_id = -1
        self.node_id = -1
        self.next_id = -1
        self.setup_node(node_id, init_role)
        self.prev_node_change_callback = prev_node_change_callback

    def handle_delete_or_change_event(self, event):
        print("event is", event)
        if event.type=='CHANGED':
            print("Node", str(self.node_id), "handle changed event")
            node_id = int(event.path[6])
            # print(self.zk.get("/base/" + str(node_id)))
            if node_id < self.node_id:
                self.prev_node_change_callback(event)
            print("after event callback")
            
        elif event.type=='DELETED':
            # get node id
            if int(event.path[6]) < self.node_id:
                # check if self is the new head
                self._get_largest_smaller
            else:
                # check if self is the new tail
                self._get_smallest_larger

    def handle_child_event(self, event):
        print("event is", event)
        if event.type == "CHILD":
            self._get_neighbors()

    def _get_largest_smaller(self):
        largest_smaller = None
        for node in self.zk.get_children("/base"):
            if int(node) < self.node_id:
                if largest_smaller is None or int(node) > largest_smaller:
                    largest_smaller = int(node)
        if largest_smaller is None:
            self.set_head()
        else:
            self.head = False
            self.prev_id = largest_smaller
            self.zk.exists("/base/"+str(self.prev_id), watch=self.handle_delete_or_change_event)

    def _get_smallest_larger(self):
        smallest_larger = None
        for node in self.zk.get_children("/base"):
            if int(node) > self.node_id:
                if smallest_larger is None or int(node) < smallest_larger:
                    smallest_larger = int(node)   
        if smallest_larger is None:
            self.set_tail()
        else: 
            self.tail = False
            self.next_id = smallest_larger
            self.zk.exists("/base/"+str(self.next_id), watch=self.handle_delete_or_change_event)
    
    def _get_neighbors(self):
        largest_smaller = None
        smallest_larger = None
        for node in self.zk.get_children("/base"):
            if int(node) < self.node_id:
                if largest_smaller is None or int(node) > largest_smaller:
                    largest_smaller = int(node)
            elif int(node) > self.node_id:
                if smallest_larger is None or int(node) < smallest_larger:
                    smallest_larger = int(node)   
        
        if largest_smaller is None:
            self.set_head()
        else:
            self.head = False
            self.prev_id = largest_smaller
            if not self.zk.exists("/base/"+str(self.prev_id), watch=self.handle_delete_or_change_event):
                # retry if there is a state change since `get_children`
                self._get_neighbors()
        
        if smallest_larger is None:
            self.set_tail()
        else:
            self.tail = False
            self.next_id = smallest_larger
            if not self.zk.exists("/base/"+str(self.next_id), watch=self.handle_delete_or_change_event):
                # retry if there is a state change since `get_children`
                self._get_neighbors()

    def setup_node(self, node_id, init_role):
        print("NODE ", node_id, " is created")
        
        self.node_id = node_id
        self.zk.create("/base/"+str(self.node_id), b"somevalue", ephemeral=True, makepath=True)

        self._get_neighbors()
        # largest_smaller = None
        # smallest_larger = None
        # for node in self.zk.get_children("/base"):
        #     if int(node) < self.node_id:
        #         if largest_smaller is None or int(node) > largest_smaller:
        #             largest_smaller = int(node)
        #     elif int(node) > self.node_id:
        #         if smallest_larger is None or int(node) < smallest_larger:
        #             smallest_larger = int(node)   
        
        # if largest_smaller is None:
        #     self.set_head()
        # else:
        #     self.head = False
        #     self.prev_id = largest_smaller
        #     self.zk.exists("/base/"+str(self.prev_id), watch=self.handle_delete_or_change_event):
        
        # if smallest_larger is None:
        #     self.set_tail()
        # else:
        #     self.tail = False
        #     self.next_id = smallest_larger
        #     self.zk.exists("/base/"+str(self.next_id), watch=self.handle_delete_or_change_event)

    def stop(self):
        self.zk.stop()
    
    def set_head(self):
        self.head = True
        self.prev_id = -1
        self.zk.get_children("/base", watch=self.handle_child_event)

    def set_tail(self):
        self.tail = True
        self.next_id = -1
        self.zk.get_children("/base", watch=self.handle_child_event)

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





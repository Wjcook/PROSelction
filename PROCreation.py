from functools import cmp_to_key
from random import randint
def comp(l, r):
    if (len(l.set) < len(r.set)):
        return -1
    elif len(l.set) > len(r.set):
        return 1
    else:
        return 0

class PROSet:
    def __init__(self, set, cost_to_go):
        self.cost_to_go = cost_to_go
        self.set = set
        self.next = []

    def get_next(pro_id):
        return self.next[pro_id]

class PROSpace:
    def __init__(self, tot):
        s = [i for i in range(tot)]
        self.nodes = []
        for i in range(1,1 << tot):
            self.nodes.append(PROSet(set([s[j] for j in range(tot) if (i & (1 << j))]), randint(1,1000)))
        self.nodes.sort(key=cmp_to_key(comp))

        l = 0
        next_start = 0
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if len(node.set) != l:
                l += 1
                if l != tot:
                    for j in range(i, len(self.nodes)):
                        if len(self.nodes[j].set) != l:
                            next_start = j
            node.next = [None for j in range(tot)]
            for j in range(next_start, len(self.nodes)):
                if len(self.nodes[j].set) == l+2:
                    break
                diff = self.nodes[j].set - node.set
                if len(diff) == 1:
                    for i in diff:
                        node.next[i] = self.nodes[j]

        def get_root_nodes(self):
            return self.nodes[:tot]





if __name__ == "__main__":
    space = PROSpace(20)

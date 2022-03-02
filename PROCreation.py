from functools import cmp_to_key
import random
from tqdm import tqdm

import graphviz


INIT_VALUE_RANGE = [0.1, 0.4]

def comp(l, r):
    if (l.set == r.set):
        return 0
    if (len(l.set) < len(r.set)):
        return -1
    elif len(l.set) > len(r.set):
        return 1
    else:
        l_iter = iter(l.set)
        r_iter = iter(r.set)
        while True:
            next_l = next(l_iter)
            next_r = next(r_iter)
            if next_l < next_r:
                return -1
            elif next_l > next_r:
                return 1

    return 0

def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid], x) < 0:
                lo = mid + 1
            else:
                hi = mid
    return lo

class PROSet:

    def __init__(self, set):
        self.set = set
        self.value = 0
        self.next= []

    def __str__(self):
        return str(self.set) + ":" + "{0:.4f}".format(self.value)

class PROSpace:
    def __init__(self, tot, display=False):
        #store best 5-10 optimal sets for a given set size
        self.tot = tot
        self.solutions = [0 for i in range(tot+1)]
        s = [i for i in range(tot)]
        self.nodes = []
        if display:
            print("Creating nodes:")
            for i in tqdm(range(1,1 << tot)):
                self.nodes.append(PROSet(set([s[j] for j in range(tot) if (i & (1 << j))])))
        else:
            for i in range(1,1 << tot):
                self.nodes.append(PROSet(set([s[j] for j in range(tot) if (i & (1 << j))])))
        self.nodes.sort(key=cmp_to_key(comp))

        l = 0
        next_start = 0
        if display:
            print("Creating edges:")
            for i in tqdm(range(len(self.nodes))):
                node = self.nodes[i]
                if (l == 0 or l == 1):
                    node.value = random.uniform(INIT_VALUE_RANGE[0], INIT_VALUE_RANGE[1])
                    if self.solutions[l] < node.value:
                        self.solutions[l] = node.value
                if len(node.set) != l:
                    l += 1
                    if l != tot:
                        for j in range(i, len(self.nodes)):
                            if len(self.nodes[j].set) != l:
                                next_start = j
                                break
                node.next = [None for j in range(tot)]
                for j in range(next_start, len(self.nodes)):
                    if len(self.nodes[j].set) == l+2:
                        break
                    diff = self.nodes[j].set - node.set
                    if len(diff) == 1:
                        for i in diff:
                            node.next[i] = j
                            self.nodes[j].value = max(random.uniform(node.value, (l+1) / self.tot), self.nodes[j].value)
                            if self.solutions[l] < self.nodes[j].value:
                                self.solutions[l] = self.nodes[j].value
        else:
            for i in range(len(self.nodes)):
                node = self.nodes[i]
                if (l == 0 or l == 1):
                    node.value = random.uniform(INIT_VALUE_RANGE[0], INIT_VALUE_RANGE[1])
                    if self.solutions[l-1] < node.value:
                        self.solutions[l-1] = node.value
                if len(node.set) != l:
                    l += 1
                    if l != tot:
                        for j in range(i, len(self.nodes)):
                            if len(self.nodes[j].set) != l:
                                next_start = j
                                break
                node.next = [None for j in range(tot)]
                for j in range(next_start, len(self.nodes)):
                    if len(self.nodes[j].set) == l+2:
                        break
                    diff = self.nodes[j].set - node.set
                    if len(diff) == 1:
                        for i in diff:
                            node.next[i] = j
                            self.nodes[j].value = max(random.uniform(node.value, (l+1) / self.tot), self.nodes[j].value)
                            if self.solutions[l-1] < self.nodes[j].value:
                                self.solutions[l-1] = self.nodes[j].value

    def visualize(self):
        dot = graphviz.Digraph()
        for i in range(len(self.nodes)):
            dot.node(str(i), str(self.nodes[i]))
            for j in range(len(self.nodes[i].next)):
                if self.nodes[i].next[j] is not None:
                    dot.edge(str(i), str(self.nodes[i].next[j]))

        # create the starting node
        dot.node("-1", "{}")
        for i in range(self.tot):
            dot.edge("-1", str(i))
        dot.render('Test_space.gv', view=False)

    def get_value(self, set):
        i = bisect_left(self.nodes, set, key=comp)
        if i < len(self.nodes):
            return self.nodes[i].value
        else:
            raise ValueError








# visualize tree exploration as well
# dot(math viz)

# if __name__ == "__main__":
#     random.seed(10)
#     space = PROSpace(5)

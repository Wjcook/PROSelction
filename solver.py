from functools import cmp_to_key
from random import randint
from PyPDF2 import PdfFileMerger


import numpy as np
from PROCreation import *


# random.seed(10)
# np.random.seed(10)
# # seed numpy random
# TOTAL_PROS = 5
# CANDIDATE_SPACE = PROSpace(TOTAL_PROS)
GRAPH_NUM = 0


class PROSetState:
    '''
    Class that represents an actual set of PROs. This class stores information
    about the actual orbits like what orbits are actually in the set, the orbits that
    can be added to the set, and the information value/cost associated with the set.
    '''

    def __init__(self, set, total_PROs, candidates):
        self.set = set
        self.total_PROs = total_PROs
        self.legal_PROs = []
        self.candidates = candidates
        for i in range(total_PROs):
            if (i not in set):
                self.legal_PROs.append(i)
    def get_legal_pros(self):

        return self.legal_PROs

    def add_PRO(self, PRO):
        new_set = self.set.copy()
        new_set.add(PRO)
        return PROSetState(new_set, self.total_PROs, self.candidates)

    def get_value(self):
        '''
        Calls the PRO integrator to actually obtain the information value/cost of
        the set of PROs.
        '''
        return self.candidates.get_value(PROSet(self.set))



class MCTSNode:
    '''
    Class that represents a node in the search tree of the Monte Carlo Solver.
    Holds information about the node in the tree like how many times the node has
    been visited, the sum of the information cost/value of all of its children nodes
    and the nodes that have yet to be explored yet.
    '''
    def __init__(self, set_state, tot, target_size, parent=None, draw_tree=False):
        self.PRO_state = set_state
        # Init the available PROs to all legal pros then as we explore more children pop
        # from this list
        self.available_PROs = self.PRO_state.get_legal_pros()
        self.children = []
        self.num_visited = 0
        self.sum_values = 0
        self.parent = parent
        self.tot = tot
        self.target_size = target_size

        self.d = draw_tree

    def draw(self, graph, c):
        if self.parent:
            graph.node(str(self.PRO_state.set), "{0}:{1}".format(str(self.PRO_state.set), \
                self.get_UCT(self.parent.num_visited, c)))
            graph.edge(str(self.parent.PRO_state.set), str(self.PRO_state.set))
        else:
            graph.node(str(self.PRO_state.set), "{0}".format(str(self.PRO_state.set)))
        for child in self.children:
            child.draw(graph, c)

    def is_terminal(self):
        # Have we explored this node yet?
        return len(self.PRO_state.set) == self.target_size

    def is_expanded(self):
        return len(self.available_PROs) == 0

    def get_UCT(self, parent_sims, c):
        if self.num_visited == 0:
            return float('NaN')
        return self.sum_values / self.num_visited + c * np.sqrt(np.log(parent_sims) / self.num_visited)

    def greedy_choice(self):
        children = [MCTSNode(self.PRO_state.add_PRO(choice), self.tot, \
            self.target_size, parent=self) for choice in self.available_PROs]

        vals = [child.PRO_state.get_value() for child in children]

        return children[np.argmax(vals)]

    def best_choice(self, c):
        '''
        Returns the child node that maximizes the Upper Confidence Bound for a certain
        value of c.
        '''
        if len(self.children) == 0:
            return self.greedy_choice()

        UCTs = [child.get_UCT(self.num_visited, c) for child in self.children]
        return self.children[np.argmax(UCTs)]

    def expand(self):
        next_pro = self.available_PROs.pop()
        next_child = MCTSNode(self.PRO_state.add_PRO(next_pro), self.tot, self.target_size, parent=self, draw_tree=self.d)
        self.children.append(next_child)
        return next_child

    def select_expand(self,c):
        cur = self

        while not cur.is_terminal():

            if not cur.is_expanded():

                res = cur.expand()
                return res
            else:
                # Continue down the treee
                cur = cur.best_choice(c)
        return cur

    def sim(self, target_set_size):
        global GRAPH_NUM

        cur_pros = self.PRO_state
        old_pros = self.PRO_state
        if self.d:
            graph = graphviz.Digraph()
            graph.node(str(cur_pros.set))

        while len(cur_pros.set) < target_set_size:
            cur_pros = cur_pros.add_PRO(np.random.choice(cur_pros.get_legal_pros()))
            if self.d:
                if len(cur_pros.set) == target_set_size:
                    graph.node(str(cur_pros.set), "{0}:{1}".format(str(cur_pros.set), \
                        cur_pros.get_value()), style='filled', fillcolor='red')
                else:
                    graph.node(str(cur_pros.set), style='filled', fillcolor='red')
                graph.edge(str(old_pros.set), str(cur_pros.set), label=str(cur_pros.set - old_pros.set))
            old_pros = cur_pros
        if self.d:
            graph.attr(label="Simulation")
            graph.render("graphs/graph{}.gv".format(GRAPH_NUM))
            GRAPH_NUM += 1
        return cur_pros.get_value()

    def back_prop(self, value):
        self.num_visited += 1
        self.sum_values += value
        if self.parent is not None:
            self.parent.back_prop(value)







class MonteCarloSolver:
    '''
    Searches a large tree of nodes that represent sets of PROs for an optimal
    subset of PROs that maximize/minimize some value.
    '''

    def __init__(self, c, iters, target_set_size, total_PROs, candidates, draw_tree=False):
        '''
        @PARAMS:
        c - exploration parameter. The larger the value the more the solver will
        prefer to explore new nodes rather than pursuing deeper into the tree

        iters - how many iterations of the solving algorithm should run

        target_set_size - The size of the subset of PROs to search for



        '''

        self.c=c
        self.iters = iters
        self.target_set_size = target_set_size
        self.solved = False
        '''
        root - is the root node of the decision space we are trying to explore
        This will be an object that will have functions to get the available actions
        Make an action and return a cost
        '''

        self.root = MCTSNode(PROSetState(set([]), total_PROs, candidates), total_PROs, target_set_size, draw_tree=draw_tree)
        self.draw = draw_tree



    def draw_tree(self, title):
        global GRAPH_NUM
        graph = graphviz.Digraph()
        self.root.draw(graph, self.c)
        graph.attr(label=title)
        graph.render("graphs/graph{}.gv".format(GRAPH_NUM))

        GRAPH_NUM += 1


    def merge_pdf(self):

        merger = PdfFileMerger()

        for i in range(GRAPH_NUM):
            merger.append("graphs/graph{}.gv.pdf".format(i))

        merger.write("result.pdf")
        merger.close()

    def solve(self):
        '''
        The actual solving algorithm. The steps of the algorithm follow the main
        steps of the MCTS algorithm: selection, expansion, simulation, and backprogation.
        '''
        for _ in range(self.iters):
            state = self.root.select_expand(self.c)
            if self.draw:
                self.draw_tree("Selection and Expansion")
            value = state.sim(self.target_set_size)
            state.back_prop(value)

        self.solved = True

        if self.draw:
            self.merge_pdf()

    def get_solution(self):
        '''
        Returns the optimal set of PROs that the solver has found so far.
        '''
        if not self.solved:
            raise ValueError("'solve' has not been called")
        cur_node = self.root
        for _ in range(self.target_set_size):
            cur_node = cur_node.best_choice(c=0.0)
        return cur_node.PRO_state

class GreedySolver:

    def __init__(self, target_set_size, total_PROs, candidates):
        self.target_size = target_set_size
        self.total_PROs = total_PROs
        self.solved = False
        self.solution = None
        self.candidates = candidates

    def solve(self):
        set_to_go = PROSetState(set(), self.total_PROs, self.candidates)
        for i in range(self.target_size):
            next_choices = [set_to_go.add_PRO(choice) for choice in set_to_go.get_legal_pros()]
            costs = [s.get_value() for s in next_choices]
            set_to_go = next_choices[np.argmax(costs)]
        self.solved = True
        self.solution = set_to_go

    def get_solution(self):
        if self.solve:
            return self.solution
        else:
            raise ValueError("'solve' has not been called")

# if __name__ == "__main__":
#
#     solver = MonteCarloSolver(0, 20, 3, TOTAL_PROS, CANDIDATE_SPACE, False)
#
#     greedy = GreedySolver(3, TOTAL_PROS, CANDIDATE_SPACE)
#     greedy.solve()
#     greedy_soln = greedy.get_solution()
#
#     solver.solve()
#     set = solver.get_solution()
#     print("{}:{}".format(set.set, set.get_value()))
#     print("{}:{}".format(greedy_soln.set, greedy_soln.get_value()))

    # CANDIDATE_SPACE.visualize()

from functools import cmp_to_key
from random import randint
from PyPDF2 import PdfFileMerger
from matplotlib.style import available

import numpy as np
# from PROCreation import *
from pro_creation.proIntUtils import cost_function
import sys
from typing import Tuple


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

    def __init__(self, set, total_PROs, states, pois):
        """Initializes a PROSetState.

        Attributes
        ----------
        set : (Python) set of int
            The ground set of PROs, stored as integers.
        total_PROs : int
            Total number of PROs.
        legal_PROs : list of int
            What PROs are able to be added (meaning they are not currently in set).
        states : list of list of list float, shape (len(set), num_time_steps, 6) 6 being 3 pos + 3 vel
            Contains states (position and velocities) of deputies during each PRO.
        cost : float
            The information cost of this set of PROs.
        pois : array of 3-vectors
            Points of interest on the object to-be-inspected.

        """
        self.set = set
        self.total_PROs = total_PROs
        self.legal_PROs = []
        self.states = states
        self.cost = None
        self.pois = pois
        for i in range(total_PROs):
            if (i not in set):
                self.legal_PROs.append(i)

    def get_legal_pros(self):
        return self.legal_PROs

    def add_PRO(self, PRO, state):
        new_set = self.set.copy()
        new_set.add(PRO)
        new_states = self.states.copy()
        new_states.append(state)
        return PROSetState(new_set, self.total_PROs, new_states, self.pois)

    def get_value(self):
        '''
        Calls the PRO integrator to actually obtain the information value/cost of
        the set of PROs.
        '''
        if self.cost is not None:
            return self.cost
        self.cost = cost_function(self.states, self.pois)
        return self.cost



class MCTSNode:
    '''
    Class that represents a node in the search tree of the Monte Carlo Solver.
    Holds information about the node in the tree like how many times the node has
    been visited, the sum of the information cost/value of all of its children nodes
    and the nodes that have yet to be explored yet.
    '''
    def __init__(self, set_state: PROSetState, target_size: int, pro_candidates: list, parent=None, draw_tree=False):
        """Initializes a MCTS node.

        Attributes
        ----------
        PRO_state : PROSetState
            What PROs are contained in this node - the ground set of the node.
        available_PROs : set of int
            What PROs are able to be added to this node's set (meaning: they are not currently in the set).
        children : list of MCTSNode
            All children that have this node as a parent.
        num_visited : int
            Number of times this node has been visited by the MCTS algorithm.
        best_value : float
            The best value encountered below this node in the tree.
        parent : MCTSNode
            The parent node of this node.
        pro_candidates : list of list of list of float, shape (size_of_PRO_library, num_time_steps, 6) 6 = 3 pos + 3 vel
            Contains states (position and velocities) of deputies during each PRO (master list).
        target_size : int
            The final number of PROs in a set (e.g. 5).
        best_state : PROSetState
            The best set contained below this node in the tree.
        d : bool
            Whether to draw the tree.
        """
        self.PRO_state = set_state
        # Init the available PROs to all legal pros then as we explore more children pop
        # from this list
        self.available_PROs = self.PRO_state.get_legal_pros()
        self.children = []
        self.num_visited = 0
        self.best_value = None
        self.sum_vals = 0 # used to compute the average value int UCT calc
        self.parent = parent
        self.pro_candidates = pro_candidates
        self.target_size = target_size
        self.best_state = None

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

    def get_UCT(self, parent_sims: int, c: float):
        if self.num_visited == 0:
            return float('NaN')
        return (self.sum_vals / self.num_visited) - c * np.sqrt(np.log(parent_sims) / self.num_visited)
        # [TODO} is there a paper about this UCT algorithm? I expect UCT to be Q/N + c*sqrt(log(parent_sims)/visits)
        #   whereas here we use the best value rather than Q/N, and there's a minus on the exploration term rather than plus.

    def greedy_choice(self):
        children = [MCTSNode(self.PRO_state.add_PRO(choice, self.pro_candidates[choice]), self.tot, \
            self.target_size, parent=self) for choice in self.available_PROs]

        vals = [child.PRO_state.get_value() for child in children]

        return children[np.argmin(vals)]

    def best_choice(self, c):
        '''
        Returns the child node that maximizes the Upper Confidence Bound for a certain
        value of c.
        '''
        if len(self.children) == 0:
            return self.greedy_choice()

        UCTs = [child.get_UCT(self.num_visited, c) for child in self.children]
        return self.children[np.argmin(UCTs)]

    def expand(self):
        rand_ind = np.random.randint(len(self.available_PROs))
        next_pro = self.available_PROs.pop(rand_ind)
        next_child = MCTSNode(self.PRO_state.add_PRO(next_pro, self.pro_candidates[next_pro]), self.target_size, self.pro_candidates, parent=self, draw_tree=self.d)
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

    def sim(self, target_set_size: int) -> Tuple[float, PROSetState]:
        global GRAPH_NUM

        cur_pros = self.PRO_state
        old_pros = self.PRO_state
        if self.d:
            graph = graphviz.Digraph()
            graph.node(str(cur_pros.set))

        while len(cur_pros.set) < target_set_size:
            r_choice = np.random.choice(cur_pros.get_legal_pros())
            cur_pros = cur_pros.add_PRO(r_choice, self.pro_candidates[r_choice])
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
        return (cur_pros.get_value(), cur_pros)

    def back_prop(self, value: float, state: PROSetState) -> None:
        self.num_visited += 1
        self.sum_vals += value
        # [TODO] do we backpropagate value too? it seems we are backpropping 'best value' instead
        # does MCTS still explore as we expect when doing UCT based on best-value rather than average value?
        if self.best_value is None or self.best_value > value:
            self.best_value = value
            self.best_state = state
        if self.parent is not None:
            self.parent.back_prop(value, state)







class MonteCarloSolver:
    '''
    Searches a large tree of nodes that represent sets of PROs for an optimal
    subset of PROs that maximize/minimize some value.
    '''

    def __init__(self, c: float, iters: int, target_set_size: int, pro_candidates: list, pois: list, seed_tree=False, draw_tree=False):
        '''
        @PARAMS:
        c - exploration parameter. The larger the value the more the solver will
        prefer to explore new nodes rather than pursuing deeper into the tree

        iters - how many iterations of the solving algorithm should run

        target_set_size - The size of the subset of PROs to search for

        seed_tree : bool
            Seeds the Monte-Carlo tree with a single greedy choice.

        '''
        self.c=c
        self.iters = iters
        self.target_set_size = target_set_size
        self.solved = False
        self.pois = pois
        '''
        root - is the root node of the decision space we are trying to explore
        This will be an object that will have functions to get the available actions
        Make an action and return a cost
        '''

        self.root = MCTSNode(PROSetState(set([]), len(pro_candidates), [], self.pois), target_set_size, pro_candidates, draw_tree=draw_tree)
        self.draw = draw_tree
        self.seed_tree = seed_tree



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
        if self.seed_tree :
            state = self.root
            for _ in range(self.iters):
                state = state.greedy_choice()
                value, sim_state = state.sim(self.target_set_size)
                state.back_prop(value, sim_state)
    
        for _ in range(self.iters):
            state = self.root.select_expand(self.c)
            if self.draw:
                self.draw_tree("Selection and Expansion")
            value, sim_state = state.sim(self.target_set_size)
            state.back_prop(value, sim_state)

        self.solved = True

        if self.draw:
            self.merge_pdf()
    
    def iter(self):
        state = self.root.select_expand(self.c)
        if self.draw:
            self.draw_tree("Selection and Expansion")
        value, sim_state = state.sim(self.target_set_size)
        state.back_prop(value, sim_state)
        self.solved = True
        return self.root.best_value

    def get_solution(self):
        '''
        Returns the optimal set of PROs that the solver has found so far.
        '''
        if not self.solved:
            raise ValueError("'solve' has not been called")
        if self.root.best_state is not None:
            return self.root.best_state

        cur_node = self.root
        for _ in range(self.target_set_size):
            cur_node = cur_node.best_choice(c=0.0)
            # This has the algorithm select the next best action by picking the highest
                # _average_ value node ('max child' choice). Another common option is 'robust child', which is picking the one with
                # the highest visit count. I wonder if there would be performance changes b/w the two.
        return cur_node.PRO_state



class GreedySolver:

    def __init__(self, target_set_size, pro_candidates, pois):
        self.target_size = target_set_size
        self.solved = False
        self.solution = None
        self.pro_candidates = pro_candidates
        self.pois = pois

    def solve(self):
        set_to_go = PROSetState(set(), len(self.pro_candidates), [], self.pois)
        for i in range(self.target_size):
            next_choices = [set_to_go.add_PRO(choice, self.pro_candidates[choice]) for choice in set_to_go.get_legal_pros()]
            costs = [s.get_value() for s in next_choices]
            set_to_go = next_choices[np.argmin(costs)]
        self.solved = True
        self.solution = set_to_go

    def get_solution(self):
        if self.solve:
            return self.solution
        else:
            raise ValueError("'solve' has not been called")

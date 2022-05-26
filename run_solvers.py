import time
from solver import MonteCarloSolver
from solver import GreedySolver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from proIntUtils import gen_poi, find_min_cost
import pickle
import yaml


PRO_DIR = "pro_spaces"
EXPERIMENT_DIR = "experiments"



class ExperimentSpace: ## to string implementation for actual experiment
    def __init__(self, exper_name, candidate_filename, pois, target_cardinality, candidate_size):
        self.exper_name = exper_name
        self.candidate_filename = candidate_filename
        self.pois = pois
        self.target_cardinality = target_cardinality
        self.candidate_size = candidate_size
        self.min_cost = None
        self.min_orbits = None

    def get_candidates(self):
        """
        Returns the candidate space as a np array with shape (num_deputies, time_steps, 6)
        """
        return np.load(self.candidate_filename)
    
    def get_global_min(self):
        if self.min_cost is not None:
            return (self.min_cost, self.min_orbits)
        space = self.get_candidates()
        print("Calculating global minimum cost:")
        self.min_cost, self.min_orbits = find_min_cost(space, self.pois, self.target_cardinality)
        return (self.min_cost, self.min_orbits)

    
        
SIMS = 1
TEST_SPACE = 16
TEST_SIZE = 8
C = 0.1
ITERS = 1000
def run_mcts(c, iters, target_size, space, pois):
    global SIMS
    tot_time = 0
    tot_val = 0
    for i in range(SIMS):
        solver = MonteCarloSolver(c, iters, target_size, space, pois, seed_tree=True)
        s = time.time()
        solver.solve()
        soln = solver.get_solution()
        e = time.time()
        tot_time += e - s
        tot_val += soln.get_value()
        del solver

    return (tot_time / SIMS, tot_val / SIMS)



def run_greedy(target_size, space, pois):
    tot_time = 0
    tot_val = 0

    solver = GreedySolver(target_size, space, pois)
    s = time.time()
    solver.solve()
    soln = solver.get_solution()
    e = time.time()
    tot_time += e - s
    tot_val += soln.get_value()
    del solver

    return (tot_time, tot_val, list(soln.set))

def sim_iters(iter_choices, target_size, space, pois, c):
    run_time = []
    perf = []
    orbs = []
    solver = MonteCarloSolver(c, 1, target_size, space, pois)
    for _ in tqdm(range(1,iter_choices)):
        s = time.time()
        val = solver.iter()
        e = time.time()
        run_time.append(e-s)
        perf.append(val)
        orbs.append(solver.get_solution().set)
    return (run_time, perf, orbs)



# ------------
# The following functions are experiments

# TODO global maximum
def exper_mcts_perf_by_iters(exper_space, iters):
    space = exper_space.get_candidates()
    min = exper_space.get_global_min()[0]

    r, p = sim_iters(iters, exper_space.target_cardinality, space, exper_space.pois)
    plt.figure(1)
    plt.plot(range(1,iters), r)
    plt.title('Runtime of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Runtime(s)")
    plt.savefig('figs/mcts_time_by_iters_for_{}.png'.format(exper_space.exper_name))
    plt.figure(2)
    plt.plot(range(1,iters), p, label='MCTS Performance')
    plt.axhline(y=min, label='Global Minimum')
    plt.title('Performance of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Value of the Optimal Set Found")
    plt.legend()
    print("Writing to " + 'figs/mcts_perf_by_iters_for_{}.png'.format(exper_space.exper_name))
    plt.savefig('figs/mcts_perf_by_iters_for_{}.png'.format(exper_space.exper_name))


def exper_mcts_vs_greedy(expers,exper_name):
    g_run = []
    g_perf = []
    m_run = []
    m_perf = []
    global_perf = []
    sizes = []
    for i in tqdm(range(len(expers))):

        space = expers[i].get_candidates()
        t, p = run_greedy(expers[i].target_cardinality, space.tolist(), expers[i].pois)
        g_run.append(t)
        g_perf.append(p)
        t, p = run_mcts(C, ITERS, expers[i].target_cardinality, space.tolist(), expers[i].pois)
        m_run.append(t)
        m_perf.append(p)
        sizes.append(expers[i].candidate_size)
        global_perf.append(expers[i].get_global_min())
    plt.figure(1)

    plt.scatter(sizes, g_run, color='green', label='Greedy Algo')
    plt.scatter(sizes, m_run, color='blue', label='Vanilla MCTS Algo')
    plt.legend()
    plt.title("Runtime Statistics of Greedy vs. MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Runtime(s)")
    print("Writing to " + 'figs/mcts_vs_greedy_runtime_stats_for_.png'.format(exper_name))
    plt.savefig('figs/mcts_vs_greedy_runtime_statsfor_{}.png'.format(exper_name))
    plt.figure(2)
    plt.scatter(sizes, g_perf, color='green', label='Greedy Algo')
    plt.scatter(sizes, m_perf, color='blue', label='Vanilla MCTS Algo')
    plt.scatter(sizes, global_perf, color='red', label='Global Max')
    plt.legend()
    plt.title("Performance of Greedy vs. MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Value of the Optimal Set Found")
    print("Writing to " + 'figs/mcts_vs_greedy_performance_stats_for_{}.png'.format(exper_name))
    plt.savefig('figs/mcts_vs_greedy_performance_stats_for_{}.png'.format(exper_name))

def write_exper(exper_space):
    with open(EXPERIMENT_DIR + "/{}.ex".format(exper_space.exper_name), 'wb') as f:
        pickle.dump(exper_space, f)

def load_exper(exper_name):
    with open(EXPERIMENT_DIR + "/{}.ex".format(exper_name), 'rb') as f:
        experiment = pickle.load(f)
    return experiment


    


if __name__ == "__main__":
    # seed with greedy(at each node perhaps)
    # save parameters for each exper
    expers = []
    exper = load_exper("27candidates_4deps_nomPois")
    exper.get_global_min()
    write_exper(exper)
    expers.append(exper)
    exper = ExperimentSpace("125candidates_4deps_nom_pois", PRO_DIR + "/125.npy", gen_poi(), 4, 125)
    exper.get_global_min()
    write_exper(exper)
    expers.append(exper)
    # exper_mcts_perf_by_iters(exper, 5000)
    exper_mcts_vs_greedy(expers, "seeding_mcts")
    # pois = gen_poi()
    # exper_mcts_vs_greedy([27], TEST_SIZE, pois)
    # exper_mcts_perf_by_iters(27, TEST_SIZE, 5000, pois)

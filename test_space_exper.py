import time
from test_space_solver import GreedySolver
from test_space_solver import MonteCarloSolver as AlteredMonteCarloSolver
import PROCreation
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SIMS = 1
TEST_SPACE = 6
TEST_SIZE = 3
C = 1.2


def run_altered_mcts(c, iters, target_size, space):
    global SIMS
    tot_time = 0
    tot_val = 0
    for i in range(SIMS):
        solver = AlteredMonteCarloSolver(c, iters, target_size, space.tot, space)
        s = time.time()
        solver.solve()
        soln = solver.get_solution()
        e = time.time()
        tot_time += e - s
        tot_val += soln.get_value()
        del solver

    return (tot_time / SIMS, tot_val / SIMS)

def run_greedy(target_size, space):
    global SIMS
    tot_time = 0
    tot_val = 0
    for i in range(SIMS):
        solver = GreedySolver(target_size, space.nodes)
        s = time.time()
        solver.solve()
        soln = solver.get_solution()
        e = time.time()
        tot_time += e - s
        tot_val += soln.get_value()
        del solver

    return (tot_time / SIMS, tot_val / SIMS)

def sim_iters(iter_choices, target_size, space):
    run_time = []
    perf = []
    solver = AlteredMonteCarloSolver(C, 1, target_size, TEST_SIZE, space)
    for _ in tqdm(iter_choices):
        s = time.time()
        val = solver.iter()
        e = time.time()
        run_time.append(e-s)
        perf.append(val)
    return (run_time, perf)

# ------------
# The following functions are experiments


def exper_a_mcts_perf_by_iters(n_space, size, iters_to_run):
    with(open("test_spaces/pro_{}.sp".format(n_space), 'rb')) as f:
        space = pickle.load(f)
    r, p = sim_iters(iters_to_run, size, space)
    p = np.array(p)
    p /= space.solutions[size]
    plt.figure(1)
    plt.plot(iters_to_run, r)
    plt.title('Runtime of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Runtime(s)")
    plt.savefig('figs/a_mcts_time_by_iters_{}_{}.png'.format(space, size))
    plt.figure(2)
    plt.plot(iters_to_run, p, label='MCTS Performance')
    plt.axhline(y=1, label='Global Minimum')
    plt.title('Performance of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Cost of the Optimal Set Found Normalized by Global Optimum")
    plt.legend()
    plt.savefig('figs/a_mcts_perf_by_iters_{}_{}.png'.format(n_space, size))

def exper_a_mcts_vs_greedy(spaces, size):
    g_run = []
    g_perf = []
    m_run = []
    m_perf = []
    global_perf = []
    for i in tqdm(spaces):

        with(open("test_spaces/pro_{}.sp".format(i), 'rb')) as f:
            space = pickle.load(f)
        t, p = run_greedy(size, space)
        g_run.append(t)
        g_perf.append(p)
        t, p = run_altered_mcts(1.2, 150, size, space)
        m_run.append(t)
        m_perf.append(p)
        global_perf.append(space.solutions[size])
    g_perf = np.array(g_perf)
    m_perf = np.array(m_perf)
    global_perf = np.array(global_perf)
    # plt.figure(1)
    # plt.scatter(spaces, g_run, color='green', label='Greedy')
    # plt.scatter(spaces, m_run, color='blue', label='MCTS')
    # plt.legend()
    # plt.title("Runtime Statistics of Greedy vs. Altered MCTS with 150 iterations")
    # plt.xlabel("Number of PROs in Candidate Space")
    # plt.ylabel("Runtime(s)")
    # plt.show()
    # plt.savefig('figs/a_mcts_vs_greedy_runtime_stats_{}.png'.format(size))
    plt.figure(2)
    plt.scatter(spaces, g_perf/global_perf, color='green', label='Greedy')
    plt.scatter(spaces, m_perf/global_perf, color='blue', label='MCTS')
    plt.scatter(spaces, global_perf, color='red', label='Global')
    plt.legend()
    plt.title("Performance of Greedy vs. Altered MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Value of the Optimal Set Found")
    plt.show()
    # plt.savefig('figs/a_mcts_vs_greedy_performance_stats_{}.png'.format(size))


if __name__ == "__main__":
    # exper_a_mcts_perf_by_iters(TEST_SPACE, TEST_SIZE, range(9000))
    # exper_mcts_vs_greedy(range(6,15), TEST_SIZE)
    exper_a_mcts_vs_greedy(range(8,16), TEST_SIZE)
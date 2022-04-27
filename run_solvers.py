import time
from solver import MonteCarloSolver
from solver import GreedySolver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from proIntUtils import gen_poi

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
        solver = MonteCarloSolver(c, iters, target_size, space, pois)
        s = time.time()
        solver.solve()
        soln = solver.get_solution()
        e = time.time()
        tot_time += e - s
        tot_val += soln.get_value()
        del solver

    return (tot_time / SIMS, tot_val / SIMS)



def run_greedy(target_size, space, pois):
    global SIMS
    tot_time = 0
    tot_val = 0
    for i in range(SIMS):
        solver = GreedySolver(target_size, space, pois)
        s = time.time()
        solver.solve()
        soln = solver.get_solution()
        e = time.time()
        tot_time += e - s
        tot_val += soln.get_value()
        del solver

    return (tot_time / SIMS, tot_val / SIMS)

def sim_iters(iter_choices, target_size, space, pois):
    run_time = []
    perf = []
    solver = MonteCarloSolver(C, 1, target_size, space, pois)
    for _ in tqdm(range(1,iter_choices)):
        s = time.time()
        val = solver.iter()
        e = time.time()
        run_time.append(e-s)
        perf.append(val)
    return (run_time, perf)



# ------------
# The following functions are experiments

# TODO global maximum
def exper_mcts_perf_by_iters(n_space, size, iters, pois):
    space = np.load("pro_spaces/{}.npy".format(n_space))

    r, p = sim_iters(iters, size, space, pois)
    plt.figure(1)
    plt.plot(range(1,iters), r)
    plt.title('Runtime of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Runtime(s)")
    plt.savefig('figs/mcts_time_by_iters_{}_{}.png'.format(n_space, size))
    plt.figure(2)
    plt.plot(range(1,iters), p, label='MCTS Performance')
    # plt.axhline(y=space.solutions[size-1], label='Global Maximum')
    plt.title('Performance of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Value of the Optimal Set Found")
    plt.legend()
    print("Writing to " + 'figs/mcts_perf_by_iters_{}_{}.png'.format(n_space, size))
    plt.savefig('figs/mcts_perf_by_iters_{}_{}.png'.format(n_space, size))


def exper_mcts_vs_greedy(spaces, size, pois):
    g_run = []
    g_perf = []
    m_run = []
    m_perf = []
    for i in tqdm(spaces):

        space = np.load("pro_spaces/{}.npy".format(i))
        t, p = run_greedy(size, space.tolist(), pois)
        g_run.append(t)
        g_perf.append(p)
        t, p = run_mcts(C, ITERS, size, space.tolist(), pois)
        m_run.append(t)
        m_perf.append(p)
        # global_perf.append(space.solutions[size])
    plt.figure(1)
    plt.scatter(spaces, g_run, color='green', label='Greedy Algo')
    plt.scatter(spaces, m_run, color='blue', label='Vanilla MCTS Algo')
    plt.legend()
    plt.title("Runtime Statistics of Greedy vs. MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Runtime(s)")
    print("Writing to " + 'figs/mcts_vs_greedy_runtime_stats{}.png'.format(size))
    plt.savefig('figs/mcts_vs_greedy_runtime_stats{}.png'.format(size))
    plt.figure(2)
    plt.scatter(spaces, g_perf, color='green', label='Greedy Algo')
    plt.scatter(spaces, m_perf, color='blue', label='Vanilla MCTS Algo')
    # plt.scatter(spaces, global_perf, color='red', label='Global Max')
    plt.legend()
    plt.title("Performance of Greedy vs. MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Value of the Optimal Set Found")
    print("Writing to " + 'figs/mcts_vs_greedy_performance_stats_{}.png'.format(size))
    plt.savefig('figs/mcts_vs_greedy_performance_stats_{}.png'.format(size))



if __name__ == "__main__":
    # seed with greedy(at each node perhaps)
    # save parameters for each exper
    pois = gen_poi()
    exper_mcts_vs_greedy([27], TEST_SIZE, pois)
    # exper_mcts_perf_by_iters(27, TEST_SIZE, 5000, pois)

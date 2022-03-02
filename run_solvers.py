import time
from solver import MonteCarloSolver
from solver import GreedySolver
from alteredSolver import MonteCarloSolver as AlteredMonteCarloSolver
import PROCreation
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

SIMS = 1
TEST_SPACE = 16
TEST_SIZE = 8
def run_mcts(c, iters, target_size, space):
    global SIMS
    tot_time = 0
    tot_val = 0
    for i in range(SIMS):
        solver = MonteCarloSolver(c, iters, target_size, space.tot, space)
        s = time.time()
        solver.solve()
        soln = solver.get_solution()
        e = time.time()
        tot_time += e - s
        tot_val += soln.get_value()
        del solver

    return (tot_time / SIMS, tot_val / SIMS)


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
        solver = GreedySolver(target_size, space.tot, space)
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
    for num_iters in tqdm(iter_choices):
        t, p = run_mcts(1.2, num_iters, target_size, space)
        run_time.append(t)
        perf.append(p)
    return (run_time, perf)

def sim_a_iters(iter_choices, target_size, space):
    run_time = []
    perf = []
    for num_iters in tqdm(iter_choices):
        t, p = run_altered_mcts(10, num_iters, target_size, space)
        run_time.append(t)
        perf.append(p)
    return (run_time, perf)


# ------------
# The following functions are experiments

def exper_mcts_perf_by_iters(n_space, size, iters_to_run):
    with(open("spaces/pro_{}.sp".format(n_space), 'rb')) as f:
        space = pickle.load(f)

    r, p = sim_iters(iters_to_run, size, space)
    plt.figure(1)
    plt.plot(iters_to_run, r)
    plt.title('Runtime of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Runtime(s)")
    plt.savefig('figs/mcts_time_by_iters_{}_{}.png'.format(n_space, size))
    plt.figure(2)
    plt.plot(iters_to_run, p, label='MCTS Performance')
    plt.axhline(y=space.solutions[size-1], label='Global Maximum')
    plt.title('Performance of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Value of the Optimal Set Found")
    plt.legend()
    plt.savefig('figs/mcts_perf_by_iters_{}_{}.png'.format(n_space, size))

def exper_a_mcts_perf_by_iters(n_space, size, iters_to_run):
    with(open("spaces/pro_{}.sp".format(n_space), 'rb')) as f:
        space = pickle.load(f)

    r, p = sim_a_iters(iters_to_run, size, space)
    plt.figure(1)
    plt.plot(iters_to_run, r)
    plt.title('Runtime of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Runtime(s)")
    plt.savefig('figs/a_mcts_time_by_iters_{}_{}.png'.format(space, size))
    plt.figure(2)
    plt.plot(iters_to_run, p, label='MCTS Performance')
    plt.axhline(y=space.solutions[size], label='Global Maximum')
    plt.title('Performance of MCTS vs. Number of Iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Value of the Optimal Set Found")
    plt.legend()
    plt.savefig('figs/a_mcts_perf_by_iters_{}_{}.png'.format(n_space, size))

def exper_mcts_vs_greedy(spaces, size):
    g_run = []
    g_perf = []
    m_run = []
    m_perf = []
    global_perf = []
    for i in tqdm(spaces):

        with(open("spaces/pro_{}.sp".format(i), 'rb')) as f:
            space = pickle.load(f)
        t, p = run_greedy(size, space)
        g_run.append(t)
        g_perf.append(p)
        t, p = run_mcts(1.2, 150, size, space)
        m_run.append(t)
        m_perf.append(p)
        global_perf.append(space.solutions[size])
    plt.figure(1)
    plt.scatter(spaces, g_run, color='green', label='Greedy Algo')
    plt.scatter(spaces, m_run, color='blue', label='Vanilla MCTS Algo')
    plt.legend()
    plt.title("Runtime Statistics of Greedy vs. MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Runtime(s)")
    plt.savefig('figs/mcts_vs_greedy_runtime_stats{}.png'.format(size))
    plt.figure(2)
    plt.scatter(spaces, g_perf, color='green', label='Greedy Algo')
    plt.scatter(spaces, m_perf, color='blue', label='Vanilla MCTS Algo')
    plt.scatter(spaces, global_perf, color='red', label='Global Max')
    plt.legend()
    plt.title("Performance of Greedy vs. MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Value of the Optimal Set Found")
    plt.savefig('figs/mcts_vs_greedy_performance_stats_{}.png'.format(size))

def exper_a_mcts_vs_greedy(spaces, size):
    g_run = []
    g_perf = []
    m_run = []
    m_perf = []
    global_perf = []
    for i in tqdm(spaces):

        with(open("spaces/pro_{}.sp".format(i), 'rb')) as f:
            space = pickle.load(f)
        t, p = run_greedy(size, space)
        g_run.append(t)
        g_perf.append(p)
        t, p = run_altered_mcts(1.2, 150, size, space)
        m_run.append(t)
        m_perf.append(p)
        global_perf.append(space.solutions[size])
    plt.figure(1)
    plt.scatter(spaces, g_run, color='green', label='Greedy Algo')
    plt.scatter(spaces, m_run, color='blue', label='Altered MCTS Algo')
    plt.legend()
    plt.title("Runtime Statistics of Greedy vs. Altered MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Runtime(s)")
    plt.savefig('figs/a_mcts_vs_greedy_runtime_stats_{}.png'.format(size))
    plt.figure(2)
    plt.scatter(spaces, g_perf, color='green', label='Greedy Algo')
    plt.scatter(spaces, m_perf, color='blue', label='Altered MCTS Algo')
    plt.scatter(spaces, global_perf, color='red', label='Global Max')
    plt.legend()
    plt.title("Performance of Greedy vs. Altered MCTS with 150 iterations")
    plt.xlabel("Number of PROs in Candidate Space")
    plt.ylabel("Value of the Optimal Set Found")
    plt.savefig('figs/a_mcts_vs_greedy_performance_stats_{}.png'.format(size))


if __name__ == "__main__":
    exper_a_mcts_perf_by_iters(TEST_SPACE, TEST_SIZE, range(300))
    # exper_mcts_vs_greedy(range(6,15), TEST_SIZE)
    # exper_a_mcts_vs_greedy(range(8,16), TEST_SIZE)

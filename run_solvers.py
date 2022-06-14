import time
from solver import MonteCarloSolver
from solver import GreedySolver
from tqdm import tqdm



PRO_DIR = "pro_spaces"
EXPERIMENT_DIR = "experiments"


    
        
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

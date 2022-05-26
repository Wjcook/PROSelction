from tqdm import tqdm
import yaml
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from proIntUtils import gen_poi, find_min_cost
from run_solvers import sim_iters, run_greedy
import csv
import time
import sys, getopt
import matplotlib.pyplot as plt


PRO_DIR = "pro_spaces_modified_cone"
DATA_DIR = 'data'
MEMO_TABLE = join(DATA_DIR, "optimums.csv")

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if os.path.isdir(path):
            pass
        else: raise

        
# ------
# YAML configuration reading and writing

# special loader with duplicate key checking
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            assert key not in mapping
            mapping.append(key)
        return super().construct_mapping(node, deep)

def get_key(yaml_data, key) :
    # Check if key shows up in the dict yaml data and handle the error by raising an exception
    if not key in yaml_data:
        no_key_given= "No \"{}\" interval specified in configuration file. Please check that a valid mode is selected. Keys are case sensitive.".format(key)
        raise Exception(no_key_given)
    else:
        return yaml_data[key]

def read_config(config_file):
    input = open(config_file,'r')
    try:
        configuration_data = yaml.load(input,UniqueKeyLoader)
    except AssertionError:
        duplicateKey = "Error duplicate keys occur in the configuration file. To avoid undefined behavior, remove duplicate keys"
        raise Exception(duplicateKey)
    try:
        spaces = get_key(configuration_data, "spaces")
    except Exception:
        spaces = [f for f in listdir(PRO_DIR) if isfile(join(PRO_DIR, f))]

    max_iters = get_key(configuration_data, "iteration_max")
    poi_method = get_key(configuration_data, "poi_method")
    seeds = get_key(configuration_data, "seeds")
    c = get_key(configuration_data, "c")
    target_size = get_key(configuration_data, "target_size")
    res = get_key(configuration_data, "resolution_factor")
    return {
        "spaces": spaces,
        "max_iters": max_iters,
        "poi_method": poi_method,
        "seeds": seeds,
        "c": c,
        'target_size': target_size,
        'res':res
    }

def get_optimum(space_name, loaded_space, res, pois, target_size, max_time, get_opts):
    "space name is the file name and the loaded space is the np array after calling np.load"
    with open(MEMO_TABLE, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['space_name'] == space_name and int(r['target_size']) == target_size and int(r['resolution']) == res:
                orbs = []
                for o in r[None]:
                    orbs.append(o)
                return (r['value'], orbs)
    if (get_opts):
        cost, orb = find_min_cost(loaded_space, pois, target_size, MAX_TIME=max_time)
    else:
        return (0, None)
    # cost = 0
    if cost == 0:
        return (0, None) # took too long to compute

    with open(MEMO_TABLE, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([space_name, target_size, res, cost] + orb)
    return (cost, orb)

def write_config(config, exper_dir):
    mkdir_p(join(DATA_DIR, exper_dir, 'data'))
    with open(join(DATA_DIR, exper_dir, 'config.yaml'), "w") as f:
        yaml.dump(config, f)


def gen_optimums(spaces, loaded_spaces, res, loaded_pois, target_size, max_time, get_opts):
    opt = []
    print("Finding Optimum Costs....")
    for i in range(len(spaces)):
        opt.append(get_optimum(spaces[i], loaded_spaces[i], res, loaded_pois, target_size, max_time, get_opts))
    return opt

def gen_data(config, max_time, exper_dir, get_opts, run_sims=True):
    loaded_spaces = []
    for s in config["spaces"]:
        loaded_spaces.append(np.load(join(PRO_DIR, s))[:, ::config['res'], :])
    if config['poi_method'] == "nom":
        loaded_pois = gen_poi()
    else:
        raise ValueError("Only the nominal poi generation method is implemented")
    opt = gen_optimums(config['spaces'], loaded_spaces, config['res'], loaded_pois, config['target_size'], max_time, get_opts)
    if not run_sims:
        return
    
    mkdir_p(join(DATA_DIR, exper_dir, 'data'))
    for j in range(len(loaded_spaces)):
        perf_file = open(join(DATA_DIR, exper_dir, 'data', config["spaces"][j] + "_perf.csv"), 'w')
        run_file = open(join(DATA_DIR, exper_dir, 'data', config["spaces"][j] + "_run.csv"), 'w')
        orb_file = open(join(DATA_DIR, exper_dir, 'data', config["spaces"][j] + "_solutions.csv"), 'w')
        perf_writer = csv.writer(perf_file, delimiter=',')
        run_writer = csv.writer(run_file, delimiter=',')
        orb_writer = csv.writer(orb_file, delimiter=',')

        perf_writer.writerow(['seed', 'optimum_cost', 'greedy_cost'] + list(range(1,config['max_iters'])))
        run_writer.writerow(['seed', 'greedy_run'] + list(range(1,config['max_iters'])))
        orb_writer.writerow(['seed'] + ['opt_'+ str(l) for l in range(config['target_size'])] + \
            ['greed_'+ str(l) for l in range(config['target_size'])] + \
            ['iter_' + str(n) + '__' + str(m) for n in range(1,config['max_iters']) for m in range(config['target_size'])])
        print("Iteration {}".format(j))
        g_r, g_p, g_o = run_greedy(config["target_size"], loaded_spaces[j], loaded_pois)

        for i in range(len(config["seeds"])):
            row_p = [config['seeds'][i], opt[j][0], g_p]
            row_r = [config['seeds'][i], g_r]
            row_o = [config['seeds'][i]]
            if (opt[j][1] is None):
                for g in range(config['target_size']):
                    row_o.append("")
            else:
                row_o.extend(opt[j][1])
            row_o.extend(g_o)

            np.random.seed(config['seeds'][i])
            r, p, o = sim_iters(config["max_iters"], config["target_size"], loaded_spaces[j], loaded_pois, config['c'])
            row_p.extend(p)
            row_r.extend(r)
            for k in range(len(o)):
                row_o.extend(o[k])
            perf_writer.writerow(row_p)
            run_writer.writerow(row_r)
            orb_writer.writerow(row_o)
        perf_file.close()
        run_file.close()
            

def main(config_file="config.yaml", run_sims=True, max_time=None, get_opts=True):
    config = read_config(config_file)
    exper_dir = time.strftime("%Y%m%d-%H%M%S")
    if run_sims:
        write_config(config, exper_dir)
    gen_data(config, max_time, exper_dir, get_opts, run_sims=run_sims)

# def get_histo(config):
#     loaded_spaces = []
#     loaded_spaces.append(np.load(join(PRO_DIR, "22_x=(-1, 1)_y=(-1, 1)_z=(-1, 1).npy"))[:, ::config['res'], :])

#     _, _, costs = find_min_cost(loaded_spaces[0], gen_poi(), 4, None)

#     plt.hist(costs)
#     plt.show()

if __name__ == "__main__":
    # If the option -t is given then it will not actually run the solvers
    # but will only generate the optimums with the max time given as an argument
    optlst, args = getopt.getopt(sys.argv[1:], "t:o", ["time=", "opt="])
    max_time = None
    run_sims = True
    get_opts = True
    for o, a in optlst:
        if o in ('-t', '--time'):
            max_time = int(a)
            run_sims = False
        elif o == '-o':
            get_opts = False
        else:
            print("WRONG ARGUMENTS")
            exit(1)
    if len(args) > 0:
        max_time = int(args[0])

    main(run_sims=run_sims, max_time=max_time, get_opts=get_opts)
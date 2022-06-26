import numpy as np
from proIntUtils import *
from proYamlUtils import *
from os import listdir, makedirs, path
from os.path import isfile, join
import sys, getopt

KE_CUTOFF = 1000 # Energy cutoff for orbits from the grid. TODO what is a good cutoff that makes sense?

def mkdir_p(p):
    try:
        makedirs(p)
    except OSError as exc: # Python >2.5
        if path.isdir(p):
            pass
        else: raise

def create_space(config, pro_dir):
    "Saves a numpy array to disk with dimensions (num_deps, time_steps, 6)"
    mkdir_p(pro_dir)
    (orbit_params, grid_config) = get_config_dict(config)
    T = orbit_params["time"] # Times that the orbit is evaluated
    for i in grid_config['num_orbits']:
        orbit_grid = construct_grid(i, grid_config['x_range'], grid_config['y_range'], grid_config['z_range'])
        orbit_params["num_deputy"] = len(orbit_grid)
        orbit_states = compute_orbit_dynamics(orbit_grid, orbit_params)
        # animation_tools(orbit_states)
        
        # Throw out any orbits whose insertion energy is too high
        pro_candidates = []
        for i in range(orbit_params["num_deputy"]):
            l = []
            for t in range(len(T)):
                l.append(orbit_states[t, (6*(i+1)):(6*(i+2))])
            pro_candidates.append(l)
        pro_candidates = np.array(pro_candidates)
        if (len(pro_candidates) == 0):
            continue
        files = [f for f in listdir(pro_dir) if isfile(join(pro_dir, f))]
        fname = "{}_x={}_y={}_z={}.npy".format(len(pro_candidates), grid_config['x_range'], grid_config['y_range'], grid_config['z_range'])
        i = 1
        while fname in files:
            fname = "{}_x={}_y={}_z={}_{}.npy".format(len(orbit_grid), grid_config['x_range'], grid_config['y_range'], grid_config['z_range'], i)
            i += 1
        # test_visi(pro_candidates[2:3], [0, 0, -0.2])
        # print(pro_candidates.shape)
        # print(find_min_cost(pro_candidates, gen_poi(), 6))
        np.save(join(pro_dir, fname), pro_candidates)

    

    


if __name__ == "__main__":
    #Pack up these and the orbit params in a tuple for multiprocessing
    # params = [[statesToEval[0],orbParams,objectiveValue,constraintValueArray]]
    # for i in range(len(statesToEval)-1):
    #     params.append([statesToEval[i+1],orbParams,objectiveValue,constraintValueArray])
            
    # params = tuple(params)

    #Now loop through and score!
    #Will do this using multi-processing to parallelize
    #Using Pool as a context manager to ensure we close out properly 
    # with Pool(processes=20) as pool:
    #     #Applies each state to the evaluate method and submits to the pool,
    #     #then waits for them to all return
    #     costs = pool.starmap(f,params)
    # print(costs)

    #Evaluate what has the lowest cost

    #Get indexes of lowest cost
    # costs = np.array(costs)

    # get_cone_mesh(5, [0,0,0])
    optlst, args = getopt.getopt(sys.argv[1:], "c:", ["config="])
    config_file = "pro_creation/example_config.yaml"
    pro_dir = None
    for o, a in optlst:
        if o in ('-c', '--config'):
            config_file = a
        else:
            print("WRONG ARGUMENTS")
            exit(1)
    if len(args) > 0:
        pro_dir = args[0]
    else:
        print("Usage: proIntCreation.py output_dir [-c config_file]")
        exit(1)

    create_space(config_file, pro_dir)
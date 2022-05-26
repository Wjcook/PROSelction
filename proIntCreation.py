from multiprocessing import Pool
from matplotlib.pyplot import grid
import numpy as np
from proIntUtils import *
from proYamlUtils import *
from os import listdir
from os.path import isfile, join

KE_CUTOFF = 1000 # Energy cutoff for orbits from the grid. TODO what is a good cutoff that makes sense?


PRO_DIR = "pro_spaces_nozero"


def create_space(config="example_config.yaml"):
    "Saves a numpy array to disk with dimensions (num_deps, time_steps, 6)"
    (orbit_params, grid_config) = get_config_dict(config)
    T = orbit_params["time"] # Times that the orbit is evaluated
    for i in range(12, 200, 4) :
        grid_config['num_orbits'] = i
        orbit_grid = construct_grid(grid_config['num_orbits'], grid_config['x_range'], grid_config['y_range'], grid_config['z_range'])
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
        files = [f for f in listdir(PRO_DIR) if isfile(join(PRO_DIR, f))]
        fname = "{}_x={}_y={}_z={}.npy".format(len(pro_candidates), grid_config['x_range'], grid_config['y_range'], grid_config['z_range'])
        i = 1
        while fname in files:
            fname = "{}_x={}_y={}_z={}_{}.npy".format(len(orbit_grid), grid_config['x_range'], grid_config['y_range'], grid_config['z_range'], i)
            i += 1
        # test_visi(pro_candidates[2:3], [0, 0, -0.2])
        # print(pro_candidates.shape)
        # print(find_min_cost(pro_candidates, gen_poi(), 6))
        np.save(join(PRO_DIR, fname), pro_candidates)

    

    


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


    create_space()
    # get_cone_mesh(5, [0,0,0])
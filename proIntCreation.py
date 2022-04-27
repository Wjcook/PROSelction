from multiprocessing import Pool
import numpy as np
from proIntUtils import *
from proYamlUtils import *

KE_CUTOFF = 1000 # Energy cutoff for orbits from the grid. TODO what is a good cutoff that makes sense?





def main(config="example_config.yaml"):
    (orbit_params, grid_config) = get_config_dict(config)
    T = orbit_params["time"] # Times that the orbit is evaluated
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
    test_visi(pro_candidates[0:4,::50], [0, 0, -0.2])

    # np.save("pro_spaces/{}.npy".format(len(orbit_grid)), pro_candidates)

    

    


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


    main()
    # get_cone_mesh(5, [0,0,0])
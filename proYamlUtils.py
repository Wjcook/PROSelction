import yaml
import numpy as np

from proIntUtils import mu, r_e, J2

def get_config_dict(config_file):
    #Open the config file for parsing (following yaml format)
    #Note, yaml has security issues with code injection, but we assume that the configuration files are always from a trusted source
    input = open(config_file,'r')
    try:
        configuration_data = yaml.load(input,UniqueKeyLoader)
    except AssertionError:
        duplicateKey = "Error duplicate keys occur in the configuration file. To avoid undefined behavior, remove duplicate keys"
        raise Exception(duplicateKey)
    
    # Get time intervals
    time_data = get_key(configuration_data, "time")
    if not "start" in time_data or not "end" in time_data or not "step" in time_data:
        missingTimeParameters = "Either the start, end or step values of the \"time\" interval are missing in the provided time interval. Keys are case sensitive."
        raise Exception(missingTimeParameters)
    time = np.arange(eval(str(time_data["start"])),eval(str(time_data["end"])),eval(str(time_data["step"])))

    #Now assign orbit parameters 
    orb_params = {"time":time, #time steps over which to evaluate
                    "altitude":get_key(configuration_data, "altitude"), #orbit altitude
                    "ecc":get_key(configuration_data,"ecc"), #orbit eccentricity
                    "inc":get_key(configuration_data,"inc"), #orbit inclination (deg)
                    "Om":get_key(configuration_data,"RAAN"), #orbit right ascension of ascending node (deg)
                    "om":get_key(configuration_data,"omega"), #orbit argument of periapsis (deg)
                    "f":get_key(configuration_data,"theta"), #orbit angle from ascending node (deg)
                    "mu":mu,  #gravitational parameter of Earth
                    "r_e":r_e,  # Earth Radius 
                    "J2":J2} #J2 Constant of Earth
    grid_config = get_key(configuration_data, "grid")
    if not "x_range" in grid_config or not "y_range" in grid_config or not "z_range" in grid_config or not "num_orbits" in grid_config:
        missingGridParameters = "Either the x y z range values or the num_orbits were not given for the grid configuration. Keys are case sensitive."
        raise Exception(missingGridParameters)
    grid_config = {
                    "x_range": eval(str(grid_config["x_range"])),
                    "y_range": eval(str(grid_config["y_range"])),
                    "z_range": eval(str(grid_config["z_range"])),
                    "num_orbits": eval(str(grid_config["num_orbits"]))
                    }
    return (orb_params, grid_config)


def eval_array(arr):
    temp = np.zeros(np.shape(arr))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            temp[i,j] = eval(arr[i,j])
    return temp

def get_key(yaml_data, key) :
    # Check if key shows up in the dict yaml data and handle the error by raising an exception
    if not key in yaml_data:
        no_key_given= "No \"{}\" interval specified in configuration file. Please check that a valid mode is selected. Keys are case sensitive.".format(key)
        raise Exception(no_key_given)
    else:
        return yaml_data[key]


# special loader with duplicate key checking
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            assert key not in mapping
            mapping.append(key)
        return super().construct_mapping(node, deep)
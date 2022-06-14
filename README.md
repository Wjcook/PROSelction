# PROSelction

# Creating PRO Search Spaces

To generate a data set you first need to create a PRO space. This represents
the search space that MCTS and the greedy algorithm will compete to find the best
set of orbits. Relevant python scripts are in the pro_creation directory.

To create a new space, first you need to set up a config file that defines the
properites of the new space. An example file is found within the pro_creation
directory. Then simply run the script proIntCreation from the parent directory as follows

python3 pro_creation/proIntCreation output_dir [-c config_file]

This will output the space(or list of spaces) as defined in the config file as numpy
arrays. The names of the files are created as follows

[number of orbits]_x=[xrange]_y=[y_range]_z=[z_range].npy

Note a number will be appended if multiple of the same file name are present.

# Running Experiments

To create a new data set run the gen_data script as follows

python3 gen_data.py [-t seconds_to_run] [-o]

The -t option skips running the solvers and simply runs the optimum algorithm on the spaces
for a certain number of seconds. The -o option skips finding the optimum and only
runs the simulations. This data will be written under the data directory under a subdirectory
with the current date and time. This directory saves the config file that was used to generate the data.

# Data Format
The data for the experiments are stored in csvs. There are 3 csvs for each space solved.
The csv file names begin with the name of the pro space used in the simulation.
The csv appended with "perf" gives cost that each of the algorithms found, as well
as the cost that the MCTS acheived for each iteration and seed.
The csv file appended with "run" gives the runtime of each of the algorithms.
The csv file appended with "solution" gives the actual orbit set that each algorithm chose
as a list of indices into the list of orbits.

# Creating Figures
The gen_figures notebook has cells and functions that can read in the csvs of each experiment
as pandas dataframes and generate the final figures.
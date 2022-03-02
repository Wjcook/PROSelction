import pickle
import sys
from PROCreation import *


def create_space(num, file_name):
    print("Creating space with {} candidates".format(num))
    space = PROSpace(int(num), display=True)

    print("Writing to file")
    with open(file_name, 'wb') as f:
        pickle.dump(space, f)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        num_pros = sys.argv[1]
        write_file = sys.argv[2]
        print("Creating space with {} candidates".format(num_pros))
        space = PROSpace(int(num_pros), display=True)
        print("Writing to file")
        with open(write_file, 'wb') as f:
            pickle.dump(space, f)
    else:
        for i in range(6, 18):
            create_space(i, "spaces/pro_{}.sp".format(i))

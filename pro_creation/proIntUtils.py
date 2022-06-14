"""
This file gives integration utilities to the pro creation script.
It gives function that integrate the PRO forward giving state as a function of time
as well as calculating the cost of a certain pro given a set of POIs.
"""

from tqdm import tqdm
import pro_lib
import yaml
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import math
import time

#Class constants (Earth parameters) can be overwritten locally or when calling the utility methods
mu = 398600.432896939164493230  #gravitational constant
r_e = 6378.136  # Earth Radius 
J2 = 0.001082627 #J2 Constant
TARGET_CRAFT_RAD = 10 / 1000 # Modeling the target spacecraft as a sphere with radius(km)

TAN_VISI_HALF_ANGLE = np.tan(np.pi / 18)
W = 10 # Estimated basic variance based on the prior model of the target spacecraft

def compute_orbit_dynamics(state,orbParams):
    """
    Method to compute the orbit dynamics from an initial formation
    position

    Parameters
    ----------
    state : array, shape(numDeputies,3)
        initial deputy spatial configurations of the
        swarm. Should be (x,y,z) of each deputy in order
    orbParams : dict
        Dictionary of the orbital parameters used to compute
        the orbit. Required values are:
            time, altitude, ecc, inc, Om, om, f,
            num_deputy, mu, r_e, J2

            These are: time steps over which to evaluate,
            revolutions considered, orbit altitude, orbit
            eccentricity, orbit inclination (deg), orbit
            right ascension of ascending node (deg), orbit
            argument of periapsis (deg), orbit Argument of Latitude
            (deg), number of deputies,
            earth gravitational parameter, radius of earth,
            earth J2

    Returns
    ---------
    orbitState : array shape(len(time),6*(num_deputy+1))
        State vector of the orbit at each time specified.
        First 6 states are the chief orbital parameters.
        Each subsequent 6 states are a deputy's relative
        state in LVLH as (x,y,z,vx,vy,vz)
    """


    #compute initial conditions for the chief and deputy
    ys = pro_lib.initial_conditions_deputy("nonlinear_correction_linearized_j2_invariant",
                                            [orbParams["altitude"],orbParams["ecc"],orbParams["inc"],orbParams["Om"],orbParams["om"],orbParams["f"],orbParams["num_deputy"]],
                                            state,orbParams["mu"],orbParams["r_e"],orbParams["J2"])

    #Integrate the relative dynamics and chief orbital elements using pro_lib's dynamics function
    orbitState  = odeint(pro_lib.dyn_chief_deputies,ys,orbParams["time"],args=(orbParams["mu"],orbParams["r_e"],orbParams["J2"],orbParams["num_deputy"]))
    return orbitState

def cost_function(state_vector, pois):
    fps = 0
    H = 0
    for s in pois:
        fps = 0
        for i in range(len(state_vector)):
            for j in range(len(state_vector[i])):
                p = state_vector[i][j][0:3]
                if is_visible(p, s):
                    fps += (1 / np.linalg.norm(p - s)**2)
        H += 1 / ( (1 / W) + fps)
    return H

def find_min_cost(state_vector, pois, orbit_cardinality, MAX_TIME=None):
    """
    Calculates the subset of orbits that achiueves the minimum cost.
    State_vector: Numpy array of shape (num_deputies, num_time_segments, 6)

    Returns a tuple of the minimum cost and the list of indices into state_vector array that achieved it
    """
    if (len(state_vector) > 35):
        return (0, None)
    indices = list(combinations(list(range(len(state_vector))), orbit_cardinality))
    min = 1 << 10
    min_indices = []
    costs = []

    s = time.time()

    for i in range(10):
        cost = cost_function(state_vector[list(indices[i])], pois)
        costs.append(cost)
        if min > cost:
            min = cost
            min_indices = list(indices[i])
    e = time.time()

    if MAX_TIME is not None and MAX_TIME < (e-s) * (len(indices) / 10):
        return (0,0)
    for i in tqdm(range(10, len(indices))):
        cost = cost_function(state_vector[list(indices[i])], pois)
        costs.append(cost)
        if min > cost:
            min = cost
            min_indices = list(indices[i])
    return (min, min_indices, costs)
        

def is_visible(p, s):
    s_hat = s / np.linalg.norm(s)
    height = np.dot(p-s, s_hat)
    if height < 0:
        return False
    rad = np.linalg.norm((p-s) - (height * s_hat))
    return abs(rad) < abs(height * TAN_VISI_HALF_ANGLE)

def eval_orbit(state,orbParams):
    """
    Method to compute the cost of a proposed initial formation 
    position over the course of the orbit
    
    Parameters
    ----------
    state : array, shape(numDeputies,3)
        initial deputy spatial configurations of the
        swarm. Should be (x,y,z) of each deputy in order
    orbParams : dict
        Dictionary of the orbital parameters used to compute
        the orbit. Required values are:
            time, altitude, ecc, inc, Om, om, f, 
            num_deputy, mu, r_e, J2

            These are: time steps over which to evaluate, 
            revolutions considered, orbit altitude, orbit 
            eccentricity, orbit inclination (deg), orbit 
            right ascension of ascending node (deg), orbit
            argument of periapsis (deg), orbit Argument of Latitude 
            (deg), number of deputies, look angle (deg), 
            earth gravitational parameter, radius of earth,
            earth J2

    Returns
    ---------
    cost : double
        Cost of the orbit associated with this initial position
    """

    # print("Started Computing Orbit Dynamics")
    #Get orbit dyanmics
    orbitState = compute_orbit_dynamics(state,orbParams)
    
    # print("Finished Computing Orbit Dynamics, starting scoring")
    return cost_function(orbitState[1:])



def animation_tools(orbitState,azim=-100, elev=43):
    """
    Helper method to animate or provide lightweight 
    visualization of the formation dynamics. Several
    optional parameters configure the type of visualization
    or animation displayed

    Parameters
    ----------
    orbitState : array, shape(len(time),6*(num_deputy+1)
        State vector of the orbit at each time specified.
        First 6 states are the chief orbital parameters.
        Each subsequent 6 states are a deputy's relative
        state in LVLH as (x,y,z,vx,vy,vz) 
    time : array
        Time at which each state is provided
    azim : double, (default=-100)
        Azimuth angle of the initial plot rendering
    elev : double, (default=43)
        Elevation angle of the initial plot rendering
    animate : Boolean, (default=False)
        Flag to animate the formation over the orbit
    frames : int, (default=None)
        If animating, how many frames to animate. 
        Default of none animates full orbit
    animationName : string, (default="animation.mp4")
        If animating, name of output file (in local
        directory). NOTE: No overwrite protection!
    animationMode : string
        Either fixed or custom, determines if user is allowed to set
        animation view angle
    sliders : boolean, (default=False)
        Flag to produce plot with interactive sliders
        for the formation over its orbit history.
    """
    

    #Plot the relative orbit tracks, at a provided or arbitrary view angle (found to work well for these visualizations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Space craft formation in J2 dynamic orbit, LVLH frame.")
    ax.set_xlabel("x, radial out from Earth (km)")
    ax.set_ylabel("y, along track (km)")
    ax.set_zlabel("z, cross track (km)")
    ax.azim = azim
    ax.elev = elev

    
    numDep = int(len(orbitState[0])/6-1)
    if numDep <=8:
        ax.set_prop_cycle('color',plt.cm.Dark2(np.linspace(0,1,numDep)))
    elif numDep <= 10:
        ax.set_prop_cycle('color',plt.cm.tab10(np.linspace(0,1,numDep)))
    elif numDep <= 20:
        ax.set_prop_cycle('color',plt.cm.tab20(np.linspace(0,1,numDep)))
    else:
        ax.set_prop_cycle('color',plt.cm.gist_rainbow(np.linspace(0,1,numDep)))
    #Loop through each deputy
    for i in range(numDep):
    
        ax.plot(orbitState[:,6*(i+1)],orbitState[:,6*(i+1)+1],orbitState[:,6*(i+1)+2])
    
    ax.plot([0],[0],[0],"ko")
    #Get sense of scale
    scale = np.max(orbitState[:,6:])
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    
    plt.show()


def construct_grid(num_orbits, x_range, y_range, z_range):
    min_x, max_x = x_range
    min_y, max_y = y_range
    min_z, max_z = z_range
    X = np.linspace(min_x, max_x, num=int(math.ceil(np.cbrt(num_orbits))))
    Y = np.linspace(min_y, max_y, num=int(math.ceil(np.cbrt(num_orbits))))
    Z = np.linspace(min_z, max_z, num=int(math.ceil(np.cbrt(num_orbits))))
    new_orbits = []
    for x in X:
        for y in Y:
            for z in Z:
                if (x > 2 * TARGET_CRAFT_RAD and y > 2 * TARGET_CRAFT_RAD and z > 2 * TARGET_CRAFT_RAD):
                    new_orbits.append([x,y,z])

    diff = len(new_orbits) - num_orbits

    if diff < 0:
        return np.array(new_orbits)
    elif diff > 0 :
        new_orbits = new_orbits[int(diff/2) + 1:]
        new_orbits = new_orbits[:(-1*int(diff - diff/2))]
    return np.array(new_orbits)

def KE(V):
    return 0.5 * np.linalg.norm(V) ** 2


def gen_poi(num_pois=6, random=False):
    """
    Generates a set of points on a sphere with radius 10 meters. This sphere
    represents the target spacecraft and the points are points of interest on
    this spacecraft that we want to inspect from the deputy sats.
    """
    pois = np.zeros((num_pois,3))
    if random:
        # Multivariate distribution is spherically symmetric thus sampling in this way is
        # uniform over the area of a sphere.
        for i in range(num_pois):

            # create rng at the beginning with seed
            xyz = np.random.default_rng().normal(0, 5.0, 3)
            # Avoid sampling close to zero to avoid numerical instability
            while np.isclose(xyz, [0,0,0]).any():
                xyz = np.random.default_rng().normal(0, 5.0, 3)
            xyz_norm = xyz / np.sqrt(np.sum(xyz ** 2))
            pois[i] = xyz_norm * TARGET_CRAFT_RAD
    else:
        pois[0] = np.array([0, 0, TARGET_CRAFT_RAD])
        pois[1] = np.array([0, 0, -TARGET_CRAFT_RAD])
        pois[2] = np.array([0, TARGET_CRAFT_RAD, 0])
        pois[3] = np.array([0, -TARGET_CRAFT_RAD, 0])
        pois[4] = np.array([TARGET_CRAFT_RAD, 0, 0])
        pois[5] = np.array([-TARGET_CRAFT_RAD, 0, 0])
    return pois


def get_cone_mesh(h, s):
    theta1 = np.linspace(0, 2*np.pi, 100)
    r1 = np.linspace(-(h*TAN_VISI_HALF_ANGLE), 0, 100)
    t1, R1 = np.meshgrid(theta1, r1)

    # point of cone at the origin
    X = R1*np.cos(t1)
    Y = R1*np.sin(t1)
    Z = R1/TAN_VISI_HALF_ANGLE
    # code to rotate the given cone below WIP

    # r = R.from_euler('x', 90, degrees=True) from rotation vector
    # newX = np.zeros(X.shape)
    # newY = np.zeros(Y.shape)
    # newZ = np.zeros(Z.shape)
    # mult = np.dot(r.as_matrix(), [np.ravel(X), np.ravel(Y), np.ravel(Z)])
    # newX = mult[0,:].reshape(X.shape)
    # newY = mult[1,:].reshape(Y.shape)
    # newZ = mult[2,:].reshape(Z.shape)
    # Translate
    X += s[0]
    Y += s[1]
    Z += s[2]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')


    # ax.set_xlabel('x axis')
    # ax.set_ylabel('y axis')
    # ax.set_zlabel('z axis')
    # ax.set_xlim(-2.5, 2.5)
    # ax.set_ylim(-2.5, 2.5)
    # ax.set_zlim(0, 5)

    # ax.plot_surface(X, Y, Z, alpha=0.8, color="blue")
    # ax.plot_surface(X, Y, Z, alpha=0.8)
    # fig. savefig ("Cone.png", dpi=100, transparent = False)

    # plt.show()
    return (X,Y,Z)
    





def test_visi(state_vector, poi, azim=-100, elev=43):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Testing Visibility from a point of interest")
    ax.set_xlabel("x, radial out from Earth (km)")
    ax.set_ylabel("y, along track (km)")
    ax.set_zlabel("z, cross track (km)")
    ax.azim = azim
    ax.elev = elev
    scale = np.max(state_vector[:,0:3])
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    visible_points = []
    for dep in state_vector:
        ax.plot(dep[:, 0], dep[:, 1], dep[:, 2])
        for p in dep[:, 0:3]:
            if is_visible(p, poi):
                # ax.scatter(p[0], p[1], p[2])
                visible_points.append(p)

    ax.scatter(poi[0], poi[1], poi[2])
    visible_points = np.array(visible_points)
    ax.plot(visible_points[:,0], visible_points[:,1], visible_points[:,2])
    (X,Y,Z) = get_cone_mesh(scale,poi)
    ax.plot_surface(X, Y, Z, alpha=0.3, color="blue")
    plt.show()
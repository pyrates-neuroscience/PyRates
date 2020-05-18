# pyrates imports
from pyrates.ir.circuit import CircuitIR
from pyrates.backend import ComputeGraph
from pyrates.utility.visualization import plot_connectivity
from pyrates.frontend import CircuitTemplate

# additional imports
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats.distributions import norm
from pandas import DataFrame
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette
from copy import deepcopy
import time

__author__ = "Richard Gast"
__status__ = "Development"


########################
# parameter definition #
########################

# base parameters
#################

N = 10                  # number of grid points along x/y axis
N2 = int(N*N)
inter_col_dist = 1e-3   # distance between cortical columns in meter
v = 4.0                 # global velocity of axonal signal propagation in meter per second
k = 10.0                # global connection weight scaling

# connectivity parameters
#########################

# initialization
weights = np.zeros((N2, N2))

# create grid to place columns on
grid_points = np.arange(0, N) * inter_col_dist
x, y = np.meshgrid(grid_points, grid_points)
grid_points = []
for i in range(N):
    idx1, idx2 = i, 0
    while idx1 >= 0:
        grid_points.append([x[idx1, idx2], y[idx1, idx2]])
        idx1 -= 1
        idx2 += 1
for i in range(N):
    idx1, idx2 = N-i-1, N-1
    while idx1 <= N-1:
        grid_points.append([x[idx1, idx2], y[idx1, idx2]])
        idx2 -= 1
        idx1 += 1
grid_points = np.asarray(grid_points)
dists = cdist(grid_points, grid_points)
plt.matshow(dists)
plt.show()

# create distance-dependent connectivity
conn_kernel = norm(0., 0.01).pdf(np.linspace)
plt.plot(conn_kernel)
plt.show()


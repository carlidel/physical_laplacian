import numpy as np
import os
import matplotlib.pyplot as plt
from network_tools import *
import genetic_algorithm as ga
import simulated_annealing as sa
from big_test import *
import pickle

# Batch on basic fitness (old style)

n_iterations = 12
value_resolution = 100
min_mass = 1
max_mass = 100

# 1D line (n = 10)
print("1D line 10")

N = 10
network_tupla = create_path_graph(N)

obtained_weights, scores = big_test_ga(network_tupla, 1, "nodes", ga.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/line_10_f0_ga.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

obtained_weights, scores = big_test_sa(network_tupla, 1, "nodes", sa.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/line_10_f0_sa.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

# 1D line (n = 25)
print("1D line 25")

N = 25
network_tupla = create_path_graph(N)

obtained_weights, scores = big_test_ga(network_tupla, 1, "nodes", ga.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/line_25_f0_ga.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

obtained_weights, scores = big_test_sa(network_tupla, 1, "nodes", sa.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/line_25_f0_sa.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)


# 2D lattice (n = 5)
print("2D lattice 5")

N = 5
network_tupla = create_lattice(N, N)

obtained_weights, scores = big_test_ga(network_tupla, 2, "nodes", ga.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/lattice_5_f0_ga.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

obtained_weights, scores = big_test_sa(network_tupla, 2, "nodes", sa.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/lattice_5_f0_sa.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

# 2D lattice (n = 10)
print("2D lattice 10")

N = 10
network_tupla = create_lattice(N, N)

obtained_weights, scores = big_test_ga(network_tupla, 2, "nodes", ga.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/lattice_10_f0_ga.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

obtained_weights, scores = big_test_sa(network_tupla, 2, "nodes", sa.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/lattice_10_f0_sa.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

# 2D lattice (n = 20)
print("2D lattice 20")

N = 20
network_tupla = create_lattice(N, N)

obtained_weights, scores = big_test_ga(network_tupla, 2, "nodes", ga.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/lattice_20_f0_ga.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

obtained_weights, scores = big_test_sa(network_tupla, 2, "nodes", sa.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/lattice_20_f0_sa.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

# 3D cube (n = 5)
print("3D cube 5")

N = 5
network_tupla = create_cube(N, N, N)

obtained_weights, scores = big_test_ga(network_tupla, 3, "nodes", ga.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/cube_5_f0_ga.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

obtained_weights, scores = big_test_sa(network_tupla, 3, "nodes", sa.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/cube_5_f0_sa.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

# 3D cube (n = 7)
print("3D cube 7")

N = 7
network_tupla = create_cube(N, N, N)

obtained_weights, scores = big_test_ga(network_tupla, 3, "nodes", ga.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/cube_7_f0_ga.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

obtained_weights, scores = big_test_sa(network_tupla, 3, "nodes", sa.fitness,
                                       n_iterations,
                                       value_resolution=value_resolution,
                                       min_mass=min_mass,
                                       max_mass=max_mass)
with open('batch/cube_7_f0_sa.pkl', 'wb') as f:
    pickle.dump([obtained_weights, scores], f)

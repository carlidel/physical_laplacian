#%%
# Manual Quick Compare 1D
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

grafo = create_path_graph(5)
a = 2.
b = 1.
c = 1000000.
mass_list = np.array([0., 0., 1., 0., 0.])
print(create_customized_laplacian_v2(grafo[0], mass_list))
quick_compare_v2(grafo, mass_list, 1, True, False, view_thet=90)


#%%
# Manual Quick Compare 2D
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

grafo = create_lattice(3, 3)
a = 2.
b = 1.
c = 1000000.
mass_list = np.array([0., 0., 0.,
                      0., 0., 0.,
                      0., 0., 0.])
quick_compare_v2(grafo, mass_list, 2, True, False, view_thet=90)


#%%
# Manual Quick Compare 2D
from network_tools import *
import numpy as np
import os

grafo = create_lattice(5, 5)
a = 2.
b = 1.
c = 1000000.
mass_list_matrix = np.array([
 [41.,59.,86.,51.,41.],
 [54., 1.,85.,12.,51.],
 [90.,91.,94.,91.,90.],
 [55., 5.,86., 4.,55.],
 [38.,55.,91.,52.,40.]])
mass_list = mass_list_matrix.flatten()
quick_compare_v1(grafo, mass_list, 2, True, False, view_thet=90)

#%%
# Video Maker 2D
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

N = 5
grafo = create_lattice(N, N)
def explorer2D_old(network, i, n_frames):
    minimum = 0.
    maximum = 10.
    values = np.asarray(np.linspace(minimum, maximum, n_frames))
    masses = np.zeros(N*N)
    masses[11] = values[i]
    return create_customized_laplacian_v2(network, masses)

movie_maker_v2(grafo, explorer2D_old, 100, 2, kabsch=True)

#%%
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

N = 10

def explorer1D(i, n_frames):
    minimum = 1.
    maximum = 10.
    values = np.asarray(np.linspace(minimum, maximum, n_frames))
    masses = np.ones(N)
    masses[1:-1] = values[i]
    return create_mod_matrix(masses)

network_tupla = create_path_graph(N)

movie_maker(network_tupla, explorer1D, 100, 1)

#%%
# Simulated Annealing
from network_tools import *
import numpy as np
from simulated_annealing import *
import os

N = 5

network_tupla = create_lattice(N, N)

result = simulated_annealing(network_tupla, 2, "edges", fitness_v3)

#%%
# Simulated Annealing
from network_tools import *
import numpy as np
from simulated_annealing import *
import os

N = 5

network_tupla = create_lattice(N, N)

result = simulated_annealing(network_tupla, 2, "nodes", fitness, min_mass=1)

#%%
# Video Maker 2D old method
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

N = 5
grafo = create_lattice(N, N)
def explorer2D(network, i, n_frames):
    minimum = 0.
    maximum = 10.
    values = np.asarray(np.linspace(minimum, maximum, n_frames))
    masses = np.zeros(N*N)
    masses[11] = values[i]
    return create_customized_laplacian_v2(network, masses)

movie_maker_v2(grafo, explorer2D, 100, 2, kabsch=True)

#%%
# Make a movie about the old method of mass analysis
import numpy as np
import os
from network_tools import *

N = 5
grafo = create_lattice(N, N)
def explorer(network, i, n_frames):
    minimum = 1.
    maximum = 10.
    values = np.asarray(np.linspace(minimum, maximum, n_frames))
    masses = [a[1] for a in list(network.degree())]
    masses[12] = values[i]
    return create_inverse_mod_matrix(masses)

movie_maker(grafo, explorer, 100, 2, kabsch=True)

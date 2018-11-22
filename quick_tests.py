# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:15:21 2018

@author: carli
"""
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os
#%%
from parameters import *
from network_tools import *
import numpy as np
from genetic_algorithm import *

N = 4

# GA parameters

grafo = create_lattice(N, N)

progression, performance = genetic_algorithm(
        grafo,
        2,
        new_fitness_v2)

performance = performance.reshape((N,N))

#%%
from network_tools import *
import numpy as np
from genetic_algorithm import *

N = 7

# GA parameters

grafo = create_path_graph(N)

progression, performance = genetic_algorithm(
        grafo,
        1,
        new_fitness_v2)

#%%
N = 25
grafo = create_path_graph(N)


for i in range(200):
    mass_list = gaussian(np.linspace(0, 100, num=N), i, 50)
    quick_compare(grafo, mass_list, 1, False, True,
                  "foo" + str(i).zfill(4) + ".jpg", 0, 90)

os.system("ffmpeg -y -i \"foo%04d.jpg\" " + "analisi" + ".mp4")

#%%
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

N = 1000
grafo = create_path_graph(N)


mass_list = np.ones(N)

mass_list = np.array([0.] + [1000. for i in range(N-2)] + [0.])

x = sorted(get_spectral_coordinates(grafo[0],
                             create_mod_matrix(mass_list),
                             1)["x"].values)
x0 = sorted(get_spectral_coordinates(grafo[0], dim=1)["x"].values)


#print(nx.laplacian_matrix(grafo[0]).todense())
#print(np.linalg.eig(nx.laplacian_matrix(grafo[0]).todense()))

print(rmsd.kabsch_rmsd(get_spectral_coordinates(grafo[0],
                             create_mod_matrix(mass_list),
                             1).values, grafo[1].values))

plt.plot(grafo[1]["x"].values)
plt.plot(x)
plt.plot(x0)
#plt.plot(x[::-1])
#plt.plot(x0[::-1])

#quick_compare(grafo, mass_list, dim=1, show=True, view_thet=0, view_phi=90)

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
from genetic_algorithm import *
import os

grafo = create_lattice(5, 5)
a = 2.
b = 1.
c = 1000000.
mass_list = np.array([0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0.,
                      0., 0., 1.1, 0., 0.,
                      0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0.,])
quick_compare_v2(grafo, mass_list, 2, True, False, view_thet=90)

#%%
# Video Maker 2D
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

N = 3
grafo = create_lattice(N, N)
def explorer2D(network, i, n_frames):
    minimum = 0.
    maximum = 10.
    values = np.asarray(np.linspace(minimum, maximum, n_frames))
    masses = np.zeros(N*N)
    masses[4] = values[i]
    return create_customized_laplacian_v2(network, masses)

movie_maker_v2(grafo, explorer2D, 100, 2, kabsch=True)

#%%
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

grafo = create_lattice(3, 3)
a = 2.
b = 3.
c = 4.
mass_list = np.array([a, b, a, 
                      b, c, b,
                      a, b, a])
quick_compare(grafo, mass_list, 2, True, False, view_thet=90)


#%%
from network_tools import *
import numpy as np
from genetic_algorithm import *
import os

N = 3
grafo = create_lattice(5, 2)
a = 1.
b = 1.
c = 1000000.
mass_list = np.array([a, b, a,
                      b, c, b,
                      a, b, a])
#quick_compare(grafo, mass_list, 2, True, False, view_thet=90)
# GA parameters
population_size = 300
elite_rate = 0.10
random_rate = 0.10
mutation_rate = 0.5
mutation_part = 0.3  # which percentage of genes mutate?
max_iterations = 50
precision = 1e-3
value_resolution = 10

elite_size = int(population_size * elite_rate)
random_size = int(population_size * random_rate)
population_half = int(population_size * 0.5)

progression, performance = genetic_algorithm(
    grafo,
    2,
    precision,
    max_iterations,
    population_size,
    population_half,
    elite_size,
    random_size,
    mutation_rate,
    mutation_part,
    value_resolution)

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
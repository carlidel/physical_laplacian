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
from network_tools import *
import numpy as np
from genetic_algorithm import *

N = 5

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

grafo = create_path_graph(N)

progression, performance = genetic_algorithm(
        grafo,
        1,
        precision,
        max_iterations,
        population_size,
        population_half,
        elite_size,
        random_size,
        mutation_rate,
        mutation_iterations,
        value_resolution)


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

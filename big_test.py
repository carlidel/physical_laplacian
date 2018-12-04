import numpy as np
import os
import matplotlib.pyplot as plt
from network_tools import *
import genetic_algorithm as ga
import simulated_annealing as sa 

"""
Functions for easy performing of big computations
"""

def big_test_ga(network_tupla,
                dim,
                n_items="nodes",
                fitness=ga.new_fitness,
                n_iterations=12,
                value_resolution=ga.value_resolution,
                min_mass=ga.min_mass,
                max_mass=ga.max_mass,
                precision=ga.precision,
                max_iterations=ga.max_iterations,
                population_size=ga.population_size,
                population_half=ga.population_half,
                elite_size=ga.elite_size,
                random_size=ga.random_size,
                mutation_rate=ga.mutation_rate,
                mutation_part=ga.mutation_part):
    scores = []
    obtained_weights = []
    for i in range(n_iterations):
        print("iteration: " + str(i) + "/" + str(n_iterations))
        result = ga.genetic_algorithm(network_tupla,
                                      dim,
                                      n_items,
                                      fitness,
                                      value_resolution,
                                      min_mass,
                                      max_mass,
                                      precision,
                                      max_iterations,
                                      population_size,
                                      population_half,
                                      elite_size,
                                      random_size,
                                      mutation_rate,
                                      mutation_part)
        score = fitness(result,
                        network_tupla[0],
                        network_tupla[1],
                        dim)
        obtained_weights.append(result)
        scores.append(score)
    return obtained_weights, scores


def big_test_sa(network_tupla,
                dim,
                n_items="node",
                fitness=sa.fitness_v1,
                n_simulations=12,
                value_resolution=sa.value_resolution,
                min_mass=sa.min_mass,
                max_mass=sa.max_mass,
                T_0=sa.T_0,
                T_1=sa.T_1,
                n_iterations=sa.n_iterations):
    scores = []
    obtained_weights = []
    for i in range(n_simulations):
        print("iteration: " + str(i) + "/" + str(n_simulations))
        result = sa.simulated_annealing(network_tupla,
                                        dim,
                                        n_items,
                                        fitness,
                                        value_resolution,
                                        min_mass,
                                        max_mass,
                                        T_0,
                                        T_1,
                                        n_iterations)
        score = fitness(network_tupla[0],
                        result,
                        network_tupla[1],
                        dim)
        obtained_weights.append(result)
        scores.append(score)
    return obtained_weights, scores





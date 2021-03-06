import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import rmsd
import random
from network_tools import *

"""
Implementation of a generic Genetic Algorithm optimizer
for optimal laplacian mass tuning.
"""
# GA parameters
population_size = 100
elite_rate = 0.10
random_rate = 0.10
mutation_rate = 0.5
mutation_part = 0.3  # which percentage of genes mutate?
max_iterations = 500
precision = 0.001

value_resolution = 10
min_mass = 0.
max_mass = 10.

elite_size = int(population_size * elite_rate)
random_size = int(population_size * random_rate)
population_half = int(population_size * 0.5)

# GA functions

def generate_random_population(dim,
                               n_vectors, 
                               value_resolution):
    '''
    Generates "n_vectors" vectors of "dim" length with random values
    uniformally distributed in random ints
    '''
    return np.random.randint(1, value_resolution, size=[n_vectors, dim])


def crossover(vec_a, vec_b):
    '''
    Executes a crossover breeding between two vectors,
    returns new crossover vector
    '''
    if(len(vec_a) != len(vec_b)):
        raise ValueError("vectors' lenght is not the same.")
    cross_point = np.random.randint(1, len(vec_a))
    return np.concatenate((vec_a[:cross_point], vec_b[cross_point:]))


def mutate(vec, percentage, value_resolution):
    '''
    Executes vector mutation for given iterations, returns mutated vector
    '''
    for i in range(int(len(vec) * percentage)):
        vec[np.random.randint(1, len(vec))] = np.random.randint(
            1,
            value_resolution)
    return vec


def new_generation(old_gen,
                   elite_size,
                   rand_size,
                   mut_rate,
                   mut_iter,
                   full_size,
                   half_size,
                   value_resolution):
    '''
    Generates a new generation with the given selection parameters.
    old_gen must be already sorted per fitness score (from max to min).
    Returns new generation vector.
    '''
    # Elite and Crossovers
    vec_elite = old_gen[:elite_size]
    vec_rand = generate_random_population(len(old_gen[0]),
                                          rand_size,
                                          value_resolution)
    vec_crossover = [crossover(old_gen[np.random.randint(0, half_size)],
                               old_gen[np.random.randint(half_size,
                                                         full_size)])
                    for i in range(elite_size + rand_size, full_size)]
    new_gen = np.concatenate((vec_elite, vec_rand, vec_crossover))
    # Mutation
    for newborn in new_gen[1:]:
        if np.random.rand() < mutation_rate:
            newborn = mutate(newborn, mut_iter, value_resolution)
    
    return new_gen


def fitness(masses,
            network,
            target_coordinates,
            dim):
    '''
    Defines the fitness score for a given individual.
    '''
    mod_matrix = create_inverse_mod_matrix(masses)
    guess_coordinates = get_spectral_coordinates(
        nx.laplacian_matrix(network).todense(),
        mod_matrix,
        dim)
    return rmsd.kabsch_rmsd(guess_coordinates.values,
                            target_coordinates.values)


def new_fitness(masses,
                network,
                target_coordinates,
                dim):
    '''
    Fitness with new possible variation of laplacian
    '''
    laplacian = create_customized_laplacian(network, masses)
    guess_coordinates = get_spectral_coordinates(laplacian, dim=dim)
    return rmsd.kabsch_rmsd(guess_coordinates.values,
                            target_coordinates.values)


def new_fitness_v2(masses,
                   network,
                   target_coordinates,
                   dim):
    '''
    Fitness with new possible variation (v2) of laplacian
    '''
    laplacian = create_customized_laplacian_v2(network, masses)
    guess_coordinates = get_spectral_coordinates(laplacian, dim=dim)
    return rmsd.kabsch_rmsd(guess_coordinates.values,
                            target_coordinates.values)


def fitness_v3(masses,
               network,
               target_coordinates,
               dim):
    laplacian = create_weighted_laplacian(network, masses)
    guess_coordinates = get_spectral_coordinates(laplacian, dim=dim)
    return rmsd.kabsch_rmsd(guess_coordinates.values,
                            target_coordinates.values)


def genetic_algorithm(G_tupla,
                      dim,
                      n_items="nodes",
                      fitness=new_fitness,
                      value_resolution=value_resolution,
                      min_mass=min_mass,
                      max_mass=max_mass,
                      precision=precision,
                      max_iterations=max_iterations,
                      population_size=population_size,
                      population_half=population_half,
                      elite_size=elite_size,
                      random_size=random_size,
                      mutation_rate=mutation_rate,
                      mutation_part=mutation_part):
    '''
    Standard implementation of Genetic Algorithm optimizer
    '''
    G = G_tupla[0]
    target_coordinates = G_tupla[1]
    if n_items == "nodes":
        population = generate_random_population(len(G.nodes()),
                                                population_size,
                                                value_resolution)
    else:
        population = generate_random_population(len(G.edges()),
                                                population_size,
                                                value_resolution)
    progression = []
    for i in range(max_iterations):
        print("Generation: {}/{}".format(i,max_iterations))
        score = []
        for individual in population:
            masses = ((max_mass - min_mass) * (individual / value_resolution)
                      + min_mass)
            score.append(fitness(masses,
                                 G,
                                 target_coordinates,
                                 dim))
        performance = list(zip(population, score))
        performance = sorted(performance, key=lambda a:a[1])
        #print([perf[1] for perf in performance])
        print("Best score: {:f}".format(performance[0][1]))
        progression.append(performance[0])
        if(progression[0][1] <= precision):
            break
        else:
            population = new_generation([a[0] for a in performance],
                                        elite_size, random_size, mutation_rate,
                                        mutation_part,
                                        population_size, population_half,
                                        value_resolution)
    return ((max_mass - min_mass) * (performance[0][0] / value_resolution) +
            min_mass)

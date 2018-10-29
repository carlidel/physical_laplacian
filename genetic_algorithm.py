"""
Implementation of a generic Genetic Algorithm optimizer
for optimal laplacian mass tuning.
"""

# Library Imports
import networkx as nx
import numpy as np
import scipy 
import pandas as pd
import matplotlib.pylab as plt
import itertools
import random
import rmsd
from scipy.sparse import csgraph as csgraph
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Set random seed
np.random.seed(42)

# Lattice Parameters
N = 20    # Lattice dimension
F = 0     # Number of removed edges
L = 0     # Number of removed links

# GA parameters
population_size = 100
elite_rate = 0.10
mutation_rate = 0.5
mutation_iterations = 150  # How many times do we iterate a mutation?
max_iterations = 200
precision = 1e-4

elite_size = int(population_size * mutation_rate)
population_half = int(population_size * 0.5)

# Network Utilities

def remove_random_nodes(G, N):
    '''
    Given a network G, removes N random nodes from G
    '''
    to_remove = random.sample(G.nodes(), N)
    G.remove_nodes_from(to_remove)


def remove_random_edges(G, N):
    '''
    Given a network G, removes N random edges from G
    '''
    to_remove = random.sample(G.edges(), N)
    G.remove_edges_from(to_remove)


def create_lattice(lenght, width, rem_nodes=0, rem_edges=0):
    '''
    Creates a standard flat network lattice placed in a 3D space.
    Last two parameters indicates how many randomly removed nodes and edges
    we want.
    Returns a tupla (network, nodes coordinates).
    '''
    G = nx.grid_graph(dim=[lenght, width])
    remove_random_nodes(G, rem_nodes)
    remove_random_edges(G, rem_edges)
    # Little trick for having easy coordinates
    coordinates = pd.DataFrame(np.asarray(G.nodes, dtype=float),
                               columns=["x", "y"])
    # Translating and scaling
    coordinates -= coordinates.mean(axis=0)
    coordinates /= np.linalg.norm(coordinates, axis=0)
    # "z" coordinate is zero
    coordinates["z"] = 0.
    return (G, coordinates)


def get_spectral_coordinates(laplacian, mod_matrix=None):
    '''
    Given a network's laplacian, returns second, third and fourth components of
    every eigenvector as (x,y,z) axis, based on the spectral representation.
    If a modulation matrix is given, a dot operation is performed.
    '''
    if mod_matrix != None:
        laplacian = np.dot(mod_matrix, laplacian.toarray())
    
    _, eigenvectors = np.linalg.eig(laplacian.todense())
    vecs = eigenvectors[:, 1:4]
    coords = pd.DataFrame(vecs, columns=["x", "y", "z"], dtype=float)

    coords -= coords.mean(axis=0)
    coords /= np.linalg.norm(coords, axis=0)
    return coords

# GA functions

def generate_random_population(dim=N, n_vectors=population_size, 
                               low_value=0.8, high_value=1.8):
    '''
    Generates "n_vectors" vectors of "dim" length with random values
    uniformally distributed in [low_value, high_value]
    '''
    return np.random.uniform(low_value, high_value, size=[n_vectors, dim])


def crossover(vec_a, vec_b):
    '''
    Executes a crossover breeding between two vectors,
    returns new crossover vector
    '''
    if(len(vec_a) != len(vec_b)):
        raise ValueError("vectors' lenght is not the same.")
    cross_point = np.random.randint(1, len(vec_a))
    return np.concatenate((vec_a[:cross_point], vec_b[cross_point:]))


def mutate(vec, iterations=mutation_iterations, mu=1.2, sigma=0.1):
    '''
    Executes vector mutation for given iterations, returns mutated vector
    '''
    for i in range(iterations):
        vec[np.random.randint(1, len(vec))] = np.random.normal(mu, sigma)
    return vec


def new_generation(old_gen, elite_size=elite_size, mut_rate=mutation_rate,
                   full=population_size, half=population_half):
    '''
    Generates a new generation with the given selection parameters.
    old_gen must be already sorted per fitness score (from max to min).
    Returns new generation vector.
    '''
    # Elite and Crossovers
    vec_elite = old_gen[:elite_size]
    vec_crossover = [crossover(old_gen[np.random.randint(0, half)],
                               old_gen[np.random.randint(half, full)])
                    for i in range(elite_size, full)]
    new_gen = np.concatenate((vec_elite, vec_crossover))
    # Mutation
    for newborn in new_gen:
        if np.random.rand() < mutation_rate:
            newborn = mutate(newborn)
    
    return new_gen


def fitness(individual, laplacian, target_coordinates):
    '''
    Defines the fitness score for a given individual.
    '''
    mod_matrix = np.diagflat(individual)
    guess_coordinates = get_spectral_coordinates(laplacian, mod_matrix)
    return rmsd.kabsch_rmsd(guess_coordinates, target_coordinates)


def genetic_algorithm(target_coordinates, G,
                      precision=precision, max_iterations=max_iterations):
    '''
    Standard implementation of Genetic Algorithm optimizer
    '''
    population = generate_random_population(len(G.nodes()))
    laplacian = nx.laplacian_matrix(G)
    
    progression = []
    for i in range(max_iterations):
        print("Generation: {}/{}".format(i,max_iterations))
        score = []
        for individual in population:
            score.append(fitness(individual, laplacian, target_coordinates))
        performance = list(zip(population, score))
        performance = sorted(performance, key=lambda a:a[1])
        print("Best score: {:f}".format(performance[0][1]))
        progression.append(performance[0])
        if(progression[0][1] <= precision):
            break
        else:
            population = new_generation([a[0] for a in performance])
    return progression, performance[0][0]

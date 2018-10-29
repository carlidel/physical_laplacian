"""
Reimplementation and analysis of new_GA.py made by Sofia Farina.
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

# Functions

def set_weights(G, w):
    '''
    G is a network, w is an array with the desired weights for the edges of G.
    '''
    edges = G.edges()
    if(len(edges) != len(w)):
        raise ValueError("array w is not same lenght of G edges list.")
    for i in range(len(edges)):
        G.add_edge(edges[i][0], edges[i][1], weight=w[i])


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


def kabsch(pt_true, pt_guessed):
    '''
    Wrapping of Kabsch algorithm, returns pt_guessed rotated into pt_true
    with Kabsch algorithm.
    '''
    # Pre-processing
    pt_true -= pt_true.mean(axis=0)
    pt_guessed -= pt_guessed.mean(axis=0)

    pt_true /= np.linalg.norm(pt_true, axis=0)
    pt_guessed /= np.linalg.norm(pt_guessed, axis=0)

    # Kabsch
    return rmsd.kabsch_rotate(pt_guessed, pt_true)


def get_coordinates(laplacian):
    '''
    Given a laplacian matrix, returns eigenvectors as (x,y) coordinates.
    In this scenario (2D) we are interest in the second and third values
    (since the first one is by default zero).
    '''
    _, eigenvectors = np.linalg.eig(laplacian.todense())
    vecs = eigenvectors[:, 1:3]
    return np.array(vecs)

# GA functions

def generate_random_population(dim=N*N, n_vectors=population_size, 
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


def fitness(laplacian, target_coordinates):
    '''
    Defines the fitness score for a given laplacian.
    '''
    guess = get_coordinates(laplacian)
    refined_guess = kabsch(target_coordinates, guess)
    return rmsd.rmsd(refined_guess, target_coordinates)


def genetic_algorithm(target_coordinates, G,
                      precision=precision, max_iterations=max_iterations):
    '''
    Standard implementation of Genetic Algorithm optimizer
    '''
    population = generate_random_population(len(G.edges()))
    progression = []

    for i in range(max_iterations):
        print("Generation: {}/{}".format(i,max_iterations))
        score = []
        for individual in population:
            set_weights(G, individual)
            laplacian = nx.laplacian_matrix(G)
            score.append(fitness(laplacian, target_coordinates))
        performance = list(zip(population, score))
        performance = sorted(performance, key=lambda a:a[1])
        print("Best score: {:f}".format(performance[0][1]))
        progression.append(performance[0])
        if(progression[0][1] <= precision):
            break
        else:
            population = new_generation([a[0] for a in performance])
    return progression, performance[0][0]

#%%
"""
What the frick happens here???
"""
# Creating standard Lattice
R = nx.grid_graph(dim=[N,N])
remove_random_nodes(R, F)
remove_random_nodes(R, L)

# Creating laplacian and computing stuff
L = nx.laplacian_matrix(R)
eigenvalues, eigenvectors = np.linalg.eigh(L.todense())
eigenvalues = np.sort(eigenvalues)

evec1 = eigenvectors[:, 1]
evec2 = eigenvectors[:, 2]
vecs = eigenvectors[:, 1:3]
eval0 = eigenvalues[0]
eval1 = eigenvalues[1]
eval2 = eigenvalues[2]
eval3 = eigenvalues[3]
guess = np.array(vecs)

#%%
target_coordinates = np.array(R.nodes(), dtype = float)
guess_coordinates = kabsch(target_coordinates, guess)

#%%
# GA casting
weights = genetic_algorithm(target_coordinates, R)

#%%
set_weights(R, weights)
new_laplacian = nx.laplacian_matrix(R)

gc = kabsch(target_coordinates, get_coordinates(new_laplacian))

#%%


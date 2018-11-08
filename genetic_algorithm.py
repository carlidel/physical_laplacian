"""
Implementation of a generic Genetic Algorithm optimizer
for optimal laplacian mass tuning.
"""

# Library Imports
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import rmsd
import random

# Set random seed
np.random.seed(42)

# Lattice Parameters
N = 5    # Lattice dimension
F = 0     # Number of removed edges
L = 0     # Number of removed links

# GA parameters
population_size = 100
elite_rate = 0.10
mutation_rate = 0.5
mutation_iterations = 150  # How many times do we iterate a mutation?
max_iterations = 20
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


def create_path_graph(lenght):
    '''
    Creates a network line placed in a 3D space.
    Returns a tupla (network, nodes coordinates).
    '''
    G = nx.grid_graph(dim=[lenght])
    coordinates = pd.DataFrame(np.asarray(G.nodes, dtype=float), columns=["x"])
    coordinates -= coordinates.mean(axis=0)
    coordinates /= np.linalg.norm(coordinates, axis=0)
    coordinates["y"] = 0.
    coordinates["z"] = 0.
    return (G, coordinates)


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


def create_cube(lenght, width, depth, rem_nodes=0, rem_edges=0):
    G = nx.grid_graph(dim=[lenght, width, depth])
    remove_random_nodes(G, rem_nodes)
    remove_random_edges(G, rem_edges)
    coordinates = pd.DataFrame(np.asarray(G.nodes, dtype=float),
                               columns=["x", "y", "z"])
    coordinates -= coordinates.mean(axis=0)
    coordinates /= np.linalg.norm(coordinates, axis=0)
    return (G, coordinates)


def get_spectral_coordinates(laplacian, mod_matrix=np.zeros(1)):
    '''
    Given a network's laplacian, returns eigenvectors associated to the second,
    third and fourth lowest eigenvalues as (x,y,z) axis, based on the spectral
    representation.
    If a modulation matrix is given, a dot operation is performed on the
    laplacian.
    '''
    if mod_matrix.any():
        laplacian = np.dot(mod_matrix, laplacian.toarray())
        val, eigenvectors = np.linalg.eig(laplacian)
    else:
        val, eigenvectors = np.linalg.eig(laplacian.todense())
    merged = (sorted(list(zip(val, eigenvectors.transpose().tolist())),
                     key=lambda k:k[0]))
    vec1 = np.asarray(merged[1][1])
    vec2 = np.asarray(merged[2][1])
    vec3 = np.asarray(merged[3][1])
    vecs = np.column_stack((vec1, vec2, vec3))
    coords = pd.DataFrame(vecs, columns=["x", "y", "z"], dtype=float)

    coords -= coords.mean(axis=0)
    coords /= np.linalg.norm(coords, axis=0)
    return coords

# GA functions

def generate_random_population(dim, n_vectors, 
                               low_value=0.1, high_value=5.0):
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


def mutate(vec, iterations, mu=1.2, sigma=0.1):
    '''
    Executes vector mutation for given iterations, returns mutated vector
    '''
    for i in range(iterations):
        vec[np.random.randint(1, len(vec))] = np.random.normal(mu, sigma)
    return vec


def new_generation(old_gen, elite_size, mut_rate, mut_iter,
                   full_size, half_size):
    '''
    Generates a new generation with the given selection parameters.
    old_gen must be already sorted per fitness score (from max to min).
    Returns new generation vector.
    '''
    # Elite and Crossovers
    vec_elite = old_gen[:elite_size]
    vec_crossover = [crossover(old_gen[np.random.randint(0, half_size)],
                               old_gen[np.random.randint(half_size,
                                                         full_size)])
                    for i in range(elite_size, full_size)]
    new_gen = np.concatenate((vec_elite, vec_crossover))
    # Mutation
    for newborn in new_gen[1:]:
        if np.random.rand() < mutation_rate:
            newborn = mutate(newborn, mut_iter)
    
    return new_gen


def fitness(individual, laplacian, target_coordinates):
    '''
    Defines the fitness score for a given individual.
    '''
    mod_matrix = np.diagflat(individual)
    guess_coordinates = get_spectral_coordinates(laplacian, mod_matrix)
    return rmsd.kabsch_rmsd(guess_coordinates.values,
                            target_coordinates.values)


def genetic_algorithm(G_tupla, precision, max_iterations,
                      pop_size, half_pop_size, elite_size, mut_rate, mut_iter):
    '''
    Standard implementation of Genetic Algorithm optimizer
    '''
    G = G_tupla[0]
    target_coordinates = G_tupla[1]
    population = generate_random_population(len(G.nodes()), pop_size)
    laplacian = nx.laplacian_matrix(G)
    
    progression = []
    for i in range(max_iterations):
        print("Generation: {}/{}".format(i,max_iterations))
        score = []
        for individual in population:
            score.append(fitness(individual, laplacian, target_coordinates))
        performance = list(zip(population, score))
        performance = sorted(performance, key=lambda a:a[1])
        #print([perf[1] for perf in performance])
        print("Best score: {:f}".format(performance[0][1]))
        progression.append(performance[0])
        if(progression[0][1] <= precision):
            break
        else:
            population = new_generation([a[0] for a in performance],
                                        elite_size, mut_rate, mut_iter,
                                        pop_size, half_pop_size)
    return progression, performance[0][0]

#%%
# Drawing functions

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plot_3d_scatter(dataset, title="", savepath="", showfig=True):
    '''
    Plot a single dataset of 3D points.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset["x"], dataset["y"], dataset["z"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title != "":
        ax.set_title(title)
    if showfig:
        ax.view_init(30, 0)
        plt.show()
    if savepath != "":
        ax.view_init(30, 0)
        plt.savefig(savepath, dpi=300)


def plot_multiple_3d_scatter(datasets, labels, title="", savepath="",
                             showfig=True):
    '''
    Plot more datasets of 3D points in a single plot (max 5).
    '''
    if len(datasets) > 5:
        raise ValueError("too many datasets")
    if len(datasets) != len(labels):
        raise ValueError("datasets and labels must have same lenght")
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    markers = [".", "x", "*", "p", "v"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(datasets)):
        ax.scatter(datasets[i]["x"], datasets[i]["y"], datasets[i]["z"],
                   color=colors[i], marker=markers[i], label=labels[i])
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title != "":
        ax.set_title(title)
    if showfig:
        ax.view_init(30, 0)
        plt.show()
    if savepath != "":
        ax.view_init(30, 0)
        plt.savefig(savepath, dpi=300)

#%%
# Creating cube
cube = create_cube(N, N, N, F, L)
plot_3d_scatter(cube[1])
plot_3d_scatter(get_spectral_coordinates(nx.laplacian_matrix(cube[0])))
_, individual = genetic_algorithm(cube, precision, max_iterations,
                                  population_size, population_half, elite_size, mutation_rate, mutation_iterations)
mod_matrix = np.diagflat(individual)
guess_coordinates = get_spectral_coordinates(
    nx.laplacian_matrix(cube[0]), mod_matrix)
plot_multiple_3d_scatter([cube[1],
                          pd.DataFrame(rmsd.kabsch_rotate(guess_coordinates,
                                                          cube[1]),
                                       columns=["x", "y", "z"],
                                       dtype=float)],
                         ["original", "genetic"])

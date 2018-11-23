import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random
import rmsd
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import networkx as nx
import os
from network_tools import *

"""
Standard implementation of a Simulated Annealing classificator
"""

# Simulated Annealing parameters

value_resolution = 100
min_mass = 0.
max_mass = 10.
T_0 = 0.001
n_iterations = 10000

# Funcitions

def first_individual(lenght, value_resolution):
    return np.random.randint(0, value_resolution, lenght)


def new_individual(individual, value_resolution):
    new_individual = []
    for element in individual:
        new_value = element + random.sample([-1, 0, 1], 1)[0]
        if new_value < 0:
            new_individual.append(0)
        elif new_value > value_resolution:
            new_individual.append(value_resolution)
        else:
            new_individual.append(new_value)
    return np.asarray(new_individual)


def fitness_v1(net, individual, target_coordinates, min_mass, max_mass, dim):
    masses = ((max_mass - min_mass) * (individual / value_resolution)
              + min_mass)
    laplacian = create_customized_laplacian(net, masses)
    guess_coordinates = get_spectral_coordinates(net,
                                                 laplacian=laplacian,
                                                 dim=dim)
    return rmsd.kabsch_rmsd(guess_coordinates.values,
                            target_coordinates.values)


def fitness_v2(net, individual, target_coordinates, min_mass, max_mass, dim):
    masses = ((max_mass - min_mass) * (individual / value_resolution)
              + min_mass)
    laplacian = create_customized_laplacian_v2(net, masses)
    guess_coordinates = get_spectral_coordinates(net,
                                                 laplacian=laplacian,
                                                 dim=dim)
    return rmsd.kabsch_rmsd(guess_coordinates.values,
                            target_coordinates.values)


def simulated_annealing(G_tupla,
                        dim,
                        fitness=fitness_v1,
                        value_resolution=value_resolution,
                        min_mass=min_mass,
                        max_mass=max_mass,
                        T_0=T_0,
                        n_iterations=n_iterations):
    G = G_tupla[0]
    target_coordinates = G_tupla[1]
    individual = first_individual(len(G.nodes()), value_resolution)
    T_list = np.linspace(T_0, 0, n_iterations, endpoint=False)
    for i in range(len(T_list)):
        # Creation and evaluation
        fit_score = fitness(G,
                            individual,
                            target_coordinates,
                            min_mass,
                            max_mass,
                            dim)
        candidate = new_individual(individual, value_resolution)
        fit_score_candidate = fitness(G,
                                      candidate,
                                      target_coordinates,
                                      min_mass,
                                      max_mass,
                                      dim)
        # Printing progress
        if i % 100 == 0:
            print(str(i) + "/" + str(n_iterations))
            print("T: " + str(T_list[i]))
            print("Fitness: " + str(fit_score))
            print("Candidate Fitness: " + str(fit_score_candidate))
        # Annealing
        if fit_score_candidate < fit_score:
            individual = candidate
        else:
            if i % 100 == 0:
                print("P_pass: " +
                  str(np.exp((fit_score - fit_score_candidate) / T_list[i])))
            if (np.random.rand()
                    < np.exp((fit_score - fit_score_candidate) / T_list[i])):
                individual = candidate
    return candidate

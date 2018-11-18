import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random
import rmsd
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import networkx as nx

"""
Wrapping of networkx, functions and visualization tools for working with
laplacian mass tuning.
"""

# Set random seed
np.random.seed(42)

# Generic Utilities

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

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


def create_mod_matrix(mass_list):
    '''
    Given a list of masses, returns the corrispective diagonal matrix for
    modulating a laplacian matrix
    '''
    return np.diagflat(mass_list)


def create_inverse_mod_matrix(mass_list):
    '''
    Given a list of masses, returns the inverse of a diagonal matrix for
    modulating a laplacian matrix
    '''
    return np.linalg.inv(create_mod_matrix(mass_list))


def get_spectral_coordinates(network, mod_matrix=np.zeros(1), dim=3):
    '''
    Given a network, returns eigenvectors associated to the second,
    third and fourth lowest eigenvalues as (x,y,z) axis, based on the spectral
    representation.

    If a modulation matrix is given, a dot operation is performed on the
    laplacian.

    Parameters
    ----------
    network : a networkx object
    
    mod_matrix : mass modulation matrix
    
    dim : choose how many dimentions to consider (must be [1,3])
    '''
    laplacian = nx.laplacian_matrix(network)
    if mod_matrix.any():
        laplacian = np.dot(mod_matrix, laplacian.toarray())
        val, eigenvectors = np.linalg.eig(laplacian)
    else:
        val, eigenvectors = np.linalg.eig(laplacian.todense())
    merged = (sorted(list(zip(val, eigenvectors.transpose().tolist())),
                     key=lambda k: k[0]))
    vec1 = np.asarray(merged[1][1])
    if dim >= 2:
        vec2 = np.asarray(merged[2][1])
        if dim == 3:
            vec3 = np.asarray(merged[3][1])
        else:
            vec3 = np.zeros(len(merged[3][1]))
    else:
        vec2 = np.zeros(len(merged[2][1]))
        vec3 = np.zeros(len(merged[3][1]))
    vecs = np.column_stack((vec1, vec2, vec3))
    vecs -= vecs.mean(axis=0)
    vecs[:, :dim] /= np.linalg.norm(vecs[:, :dim], axis=0)
    coords = pd.DataFrame(vecs, columns=["x", "y", "z"], dtype=float)
    #print(coords)
    return coords

# Drawing and comparison functions


def plot_3d_scatter(dataset,
                    title="",
                    savepath="",
                    showfig=True,
                    view_thet=30,
                    view_phi=0):
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
        ax.view_init(view_thet, view_phi)
        plt.show()
    if savepath != "":
        ax.view_init(view_thet, view_phi)
        plt.savefig(savepath, dpi=300)


def plot_multiple_3d_scatter(datasets,
                             labels,
                             title="",
                             savepath="",
                             showfig=True,
                             view_thet=30,
                             view_phi=0):
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
        ax.view_init(view_thet, view_phi)
        plt.show()
    if savepath != "":
        ax.view_init(view_thet, view_phi)
        plt.savefig(savepath, dpi=300)


def quick_compare(network_tupla,
                  mass_list,
                  dim=3,
                  show=True,
                  save=False,
                  namefile="",
                  view_thet=30,
                  view_phi=0):
    original_coords = network_tupla[1]
    base_coords = get_spectral_coordinates(network_tupla[0], dim=dim)
    after_coords = get_spectral_coordinates(
        network_tupla[0],
        mod_matrix=create_mod_matrix(mass_list),
        dim=dim)
    print("RMSD before mass modulation: {:f}".format(rmsd.kabsch_rmsd(
        original_coords.values, base_coords.values)))
    print("RMSD after mass modulation: {:f}".format(rmsd.kabsch_rmsd(
        original_coords.values, after_coords.values)))
    if show:
        plot_multiple_3d_scatter(
            [original_coords, base_coords, after_coords],
            ["Original", "Spectral basic", "Spectral modulated"],
            view_thet=view_thet,
            view_phi=view_phi)
    if save:
        plot_multiple_3d_scatter(
            [original_coords, base_coords, after_coords],
            ["Original", "Spectral basic", "Spectral modulated"],
            view_phi=view_phi,
            view_thet=view_thet,
            showfig=False,
            savepath=namefile)

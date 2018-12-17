import pickle
import os
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import network_tools as nt
import rmsd
import matplotlib.pyplot as plt

"""
This script takes the processed .pdb files as DataFrames and returns the
Contact Map, as well as the proper network object of the protein and the
coordinates as a netowrk_tupla
"""

AA_LIST = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
           "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
           "TYR", "VAL"]


def unload_pickle_file(filepath):
    """
    Unloads a Pickle file made by atom_parser into two lists.
    (protein_name and protein_data)
    """
    with open(filepath, 'rb') as f:
        dictionary = pickle.load(f)
    protein_name = []
    protein_data = []
    for key in dictionary:
        protein_name.append(key)
        protein_data.append(dictionary[key])
    return protein_name, protein_data


def process_distance_matrix_CA(protein_name, dataset):
    filtered_dataset = filter_dataset_CA(dataset)
    N = len(filtered_dataset)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        print(str(i) + "/" + str(N))
        for j in range(i + 1, N):
            a = np.array([filtered_dataset.iloc[i]["x"],
                          filtered_dataset.iloc[i]["y"],
                          filtered_dataset.iloc[i]["z"]])
            b = np.array([filtered_dataset.iloc[j]["x"],
                          filtered_dataset.iloc[j]["y"],
                          filtered_dataset.iloc[j]["z"]])
            dist_matrix[i][j] = np.linalg.norm(a - b)
            dist_matrix[j][i] = dist_matrix[i][j]
    with open("pdb_files/" + protein_name + "_dist.pkl", "wb") as f:
        pickle.dump(dist_matrix, f)


def unload_distance_matrix_CA(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def unload_all_distance_matrix_CA(namelist, folderpath="pdb_files/"):
    distance_matrix_list = []
    for name in namelist:
        with open(folderpath + name + "_dist.pkl", 'rb') as f:
            distance_matrix_list.append(pickle.load(f))
    return distance_matrix_list


def filter_dataset_CA(dataset):
    """
    Filter only the CA atoms from a given dataset.
    Returns the list of datasets. 
    """
    return dataset[dataset["atom_name"] == "CA"]


def make_network_from_distance_matrix(distance_matrix, threshold):
    return nx.from_numpy_array(distance_matrix <= threshold)


def make_coordinate_dataset(dataset):
    """
    Returns only the coordinates of a dataset.
    """
    coordinates = dataset[["x", "y", "z"]].values
    coordinates -= coordinates.mean(axis=0)
    coordinates /= np.linalg.norm(coordinates, axis=0)
    return pd.DataFrame(coordinates,
                        columns=("x", "y", "z"))


def refresh_network_weights(dataset_list, network_list, aa_contact_map):
    for i in range(len(dataset_list)):
        edge_list = list(network_list[i].edges())
        for j in range(len(edge_list)):
            # AA identification
            label_a = dataset_list[i].iloc[edge_list[j][0]]["residue_name"]
            label_b = dataset_list[i].iloc[edge_list[j][1]]["residue_name"]
            # AA CM location
            index_a = AA_LIST.index(label_a)
            index_b = AA_LIST.index(label_b)
            # Change weigth
            network_list[i][edge_list[j][0]][edge_list[j][1]]["weigth"] = (
                aa_contact_map[index_a][index_b])
    return network_list


def create_AA_contact_map(weight_list):
    N = len(AA_LIST)
    combo_list = list(itertools.combinations(range(N), 2))
    assert len(weight_list) == len(combo_list)
    aa_contact_map = np.zeros((N, N))
    for i in range(len(weight_list)):
        aa_contact_map[combo_list[i][0]][combo_list[i][1]] = weight_list[i]
        aa_contact_map[combo_list[i][1]][combo_list[i][0]] = weight_list[i]
    return aa_contact_map


def plot_distance_statistics(distance_matrix_list,
                             n_bins,
                             title="",
                             savepath="",
                             showfig=True):
    distance_array_list = []
    for matrix in distance_matrix_list:
        distance_array_list.append(matrix.flatten())
    distance_matrix_list = np.concatenate(distance_array_list).ravel()
    plt.hist(distance_matrix_list, bins=n_bins, density=True)
    plt.xlabel("Distanza $[\\AA]$")
    plt.ylabel("Distribuzione di probabilità")
    if title != "":
        plt.title(title)
    if showfig:
        plt.show()
    if savepath != "":
        plt.savefig(savepath, dpi=300)
        plt.clf()
    

def plot_protein_network(network,
                         distance_matrix,
                         threshold,
                         coords_original,
                         coords_modified=np.zeros(1),
                         spectral_basic=False,
                         title="",
                         savepath="",
                         showfig=True,
                         view_thet=30,
                         view_phi=30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Make network to be plotted (low threshold)
    plt_network = make_network_from_distance_matrix(distance_matrix, threshold)
    # Plot original coords
    ax.scatter(coords_original["x"],
               coords_original["y"],
               coords_original["z"],
               label="Originale", c="C0")
    for edge in list(plt_network.edges):
        ax.plot((coords_original.iloc[edge[0]]["x"],
                 coords_original.iloc[edge[1]]["x"]),
                (coords_original.iloc[edge[0]]["y"],
                 coords_original.iloc[edge[1]]["y"]),
                (coords_original.iloc[edge[0]]["z"],
                 coords_original.iloc[edge[1]]["z"]),
                c="grey", alpha=0.7)
    # If any, plot given coords
    if coords_modified.any():
        ax.scatter(coords_modified["x"],
                   coords_modified["y"],
                   coords_modified["z"],
                   label="Spectral Drawing Perturbato", c="C1")
        for edge in list(plt_network.edges):
            ax.plot((coords_modified.iloc[edge[0]]["x"],
                     coords_modified.iloc[edge[1]]["x"]),
                    (coords_modified.iloc[edge[0]]["y"],
                     coords_modified.iloc[edge[1]]["y"]),
                    (coords_modified.iloc[edge[0]]["z"],
                     coords_modified.iloc[edge[1]]["z"]),
                    c="red", alpha=0.4)
    # Do you also want the spectral basic?
    if spectral_basic:
        coords_basic = nt.get_spectral_coordinates(
            nx.laplacian_matrix(network).todense(), dim=3)
        coords_basic = pd.DataFrame(
            rmsd.kabsch_rotate(coords_basic.values, coords_original.values),
            columns=["x", "y", "z"])
        ax.scatter(coords_basic["x"],
                   coords_basic["y"],
                   coords_basic["z"],
                   label="Spectral Drawing Originale", c="C2")
        for edge in list(plt_network.edges):
            ax.plot((coords_basic.iloc[edge[0]]["x"],
                     coords_basic.iloc[edge[1]]["x"]),
                    (coords_basic.iloc[edge[0]]["y"],
                     coords_basic.iloc[edge[1]]["y"]),
                    (coords_basic.iloc[edge[0]]["z"],
                     coords_basic.iloc[edge[1]]["z"]),
                    c="green", alpha=0.4)
    ax.legend()
    ax.set_xlabel("X $[\\AA]$")
    ax.set_ylabel("Y $[\\AA]$")
    ax.set_zlabel("Z $[\\AA]$")
    if title != "":
        ax.set_title(title)
    if showfig:
        ax.view_init(view_thet, view_phi)
        plt.show()
    if savepath != "":
        ax.view_init(view_thet, view_phi)
        plt.savefig(savepath, dpi=300)
        plt.clf()
    

def rotational_protein_movie(network,
                             distance_matrix,
                             threshold,
                             coords_original,
                             coords_modified=np.zeros(1),
                             spectral_basic=False,
                             title="",
                             filename="",
                             showfig=True,
                             view_phi=30,
                             n_frames=360):
    os.system("mkdir foo")
    os.system("del \"foo\\foo*.jpg\"")
    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        print(str(i) + "/" + str(n_frames))
        plot_protein_network(network,
                             distance_matrix,
                             threshold,
                             coords_original,
                             coords_modified,
                             spectral_basic,
                             title,
                             ("foo\\foo" + str(i).zfill(5) + ".jpg"),
                             view_phi=30,
                             view_thet=angle)
    os.system("ffmpeg -y -i \"foo\\foo%05d.jpg\" "
              + "img\\" + filename + ".mp4")


#%%
# Load all the data!
protein_name_list, protein_data_list = (
    unload_pickle_file("pdb_files/proteins.pkl"))
distance_matrix_CA_list = (unload_all_distance_matrix_CA(protein_name_list))

coordinate_list = []
for protein_data in protein_data_list:
    coordinate_list.append(
        make_coordinate_dataset(filter_dataset_CA(protein_data)))

#%%
plot_distance_statistics(distance_matrix_CA_list, 1000)

#%%

network = make_network_from_distance_matrix(
        distance_matrix_CA_list[0], 12.)

plot_protein_network(network,
                     distance_matrix_CA_list[0],
                     4.0,
                     coordinate_list[0],
                     coords_modified=np.zeros(1),
                     spectral_basic=True,
                     title="",
                     savepath="",
                     showfig=True,
                     view_thet=30,
                     view_phi=30)

#%%

"""
THRESHOLD = 10.0
protein_name_list, protein_data_list = (
    unload_pickle_file("pdb_files/proteins.pkl"))
protein_data_filtered = []
contact_map_list = []
network_list = []
# Will take A LOT of time... what can I do?
for dataset in protein_data_list[0:1]:
    protein_data_filtered.append(filter_dataset_CA(dataset))
    temp1, temp2 = make_protein_network_CA(dataset, THRESHOLD)
    contact_map_list.append(temp1)
    network_list.append(temp2)
    
#%%
plt.imshow(culo)
plt.colorbar()
plt.show()
"""

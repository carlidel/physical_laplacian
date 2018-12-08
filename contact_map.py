import pickle
import os
import numpy as np
import pandas as pd
import networkx as nx
import itertools

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


def filter_dataset_CA(dataset):
    """
    Filter only the CA atoms from a given dataset.
    Returns the list of datasets. 
    """
    return dataset[dataset["atom_name"] == "CA"]


def make_contact_map_CA(dataset, threshold):
    """
    Make a CA only contact map.
    Returns a list of contact maps.
    """
    filtered_dataset = filter_dataset_CA(dataset)
    N = len(filtered_dataset)
    contact_map = np.zeros((N, N), dtype=np.bool)
    for i in range(N):
        for j in range(i + 1, N):
            a = np.array([filtered_dataset.iloc[i]["x"],
                            filtered_dataset.iloc[i]["y"],
                            filtered_dataset.iloc[i]["z"]])
            b = np.array([filtered_dataset.iloc[j]["x"],
                            filtered_dataset.iloc[j]["y"],
                            filtered_dataset.iloc[j]["z"]])
            # Some skimming is needed
            if threshold >= 1:
                if threshold > np.max(np.abs(a - b)):
                    norm = np.linalg.norm(a - b)
                    print(i, j, norm)
                    if norm < threshold:
                        contact_map[i][j] = 1
                        contact_map[j][i] = 1
            else:
                norm = np.linalg.norm(a - b)
                print(i, j, norm)
                if norm < threshold:
                    contact_map[i][j] = 1
                    contact_map[j][i] = 1
    return contact_map


def make_coordinate_dataset(dataset):
    """
    Returns only the coordinates of a dataset.
    """
    coordinates = dataset[["x", "y", "z"]].values
    coordinates -= coordinates.mean(axis=0)
    coordinates /= np.linalg.norm(coordinates, axis=0)
    return pd.DataFrame(coordinates,
                        columns=("x", "y", "z"))


def make_protein_network_CA(dataset, threshold):
    contact_map = make_contact_map_CA(dataset, threshold)
    return contact_map, nx.from_numpy_array(contact_map)


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

#%%
THRESHOLD = 2.0
protein_name_list, protein_data_list = (
    unload_pickle_file("pdb_files/proteins.pkl"))
protein_data_filtered = []
contact_map_list = []
network_list = []
# Will take A LOT of time... what can I do?
for dataset in protein_data_list:
    protein_data_filtered.append(filter_dataset_CA(dataset))
    temp1, temp2 = make_protein_network_CA(dataset, THRESHOLD)
    contact_map_list.append(temp1)
    network_list.append(temp2)

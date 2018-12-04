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
    Are lists better than dictionaries in this context?
    """
    with open(filepath, 'rb') as f:
        dictionary = pickle.load(f)
    protein_name = []
    protein_data = []
    for key in dictionary:
        protein_name.append(key)
        protein_data.append(dictionary[key])
    return protein_name, protein_data


def filter_dataset_CA(dataset_list):
    """
    Filter only the CA atoms
    """
    new_dataset_list = []
    for dataset in dataset_list:
        new_dataset_list.append(dataset[dataset["atom_name"] == "CA"])
    return new_dataset_list


def make_contact_map_CA(dataset_list, threshold):
    """
    Make a CA only contact map
    """
    contact_map_list = []
    for dataset in dataset_list:
        filtered_dataset = filter_dataset_CA(dataset)
        N = len(filtered_dataset)
        contact_map = np.zeros((N, N), dtype=np.bool)
        for i in range(N):
            for j in range(i + 1, N):
                a = np.array([filtered_dataset[i]["x"],
                              filtered_dataset[i]["y"],
                              filtered_dataset[i]["z"]])
                b = np.array([filtered_dataset[j]["x"],
                              filtered_dataset[j]["y"],
                              filtered_dataset[j]["z"]])
                norm = np.linalg.norm(a - b)
                if norm < threshold:
                    contact_map[i][j] = 1
                    contact_map[j][i] = 1
        contact_map_list.append(contact_map)
    return contact_map_list


def make_coordinate_dataset_CA(dataset_list):
    coordinate_dataset_list = []
    filtered_datasets = filter_dataset_CA(dataset_list)
    for dataset in filtered_datasets:
        coordinate_dataset_list.append(
            pd.DataFrame(dataset[["x", "y", "z"]].values,
                         columns=("x", "y", "z")))
    return coordinate_dataset_list


def make_protein_network(dataset_list, threshold):
    network_list = []
    contact_map_list = make_contact_map_CA(dataset_list, threshold)
    for contact_map in contact_map_list:
        network_list.append(nx.from_numpy_array(contact_map))
    return contact_map_list, network_list


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

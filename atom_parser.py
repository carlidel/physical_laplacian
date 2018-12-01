import numpy as np
import pandas as pd
import sys
import os
import pickle

"""
Simple .pdb parser for collecting spatial info of atoms and such. 
"""

def parse_pdb_file(filepath):
    print(filepath)
    lines = open(filepath, 'r').readlines()
    atom_name = []
    residue_name = []
    residue_number = []
    x_coord = []
    y_coord = []
    z_coord = []
    element = []
    for line in lines:
        if line[0:7].strip() == "ATOM":
            atom_name.append(line[13:17].strip())
            residue_name.append(line[17:21].strip())
            residue_number.append(int(line[23:26].strip()))
            x_coord.append(float(line[31:39].strip()))
            y_coord.append(float(line[39:47].strip()))
            z_coord.append(float(line[47:55].strip()))
            element.append(line[77:79].strip())
    protein = pd.DataFrame({
        'atom_name': atom_name,
        'residue_name': residue_name,
        'residue_number': residue_number,
        'x': x_coord,
        'y': y_coord,
        'z': z_coord,
        'element': element})
    return protein

if __name__ == "__main__":
    if len(sys.argv) == 1:
        directory = "."
    else:
        directory = sys.argv[1]
    items = os.listdir(directory)
    files = []
    for name in items:
        if name.endswith(".pdb"):
            files.append(name)
    proteins = {}
    for name in files:
        protein = parse_pdb_file(directory + "/" + name)
        proteins[name[:-4]] = protein
    with open(directory + "/proteins.pkl", 'wb') as f:
        pickle.dump(proteins, f)
        



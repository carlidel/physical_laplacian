import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import network_tools as nt
import networkx as nx
import itertools
import matplotlib
import rmsd

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
DPI = 300


def print_score_block(batch_list, title, savepath):
    labels = ["$f_0$\n$GA$", "$f_0$\n$SA$", "$f_1$\n$GA$",
              "$f_1$\n$SA$", "$f_2$\n$GA$", "$f_2$\n$SA$"]
    average = []
    sigma = []
    best = []
    scores_list = [a[1] for a in batch_list]
    for scores in scores_list:
        average.append(np.average(scores))
        sigma.append(np.std(scores))
        best.append(np.min(scores))
    index = np.arange(6)
    plt.errorbar(index, average, yerr=sigma, markersize=1.5,
                 linewidth=0, elinewidth=1, marker='o', capsize=3,
                 c="C0", label="Media")
    plt.scatter(index, best, c="C1", marker='*', s=10,
                label="Miglior punteggio")
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Metodo di perturbazione ed algoritmo utilizzato")
    plt.xticks(index, labels)
    plt.ylabel("Performance $[$RMSD value$]$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()
    return average, sigma


def print_score_single(batch_list, title, savepath):
    labels = ["$f_0$\n$GA$", "$f_0$\n$SA$", "$f_1$\n$GA$",
              "$f_1$\n$SA$", "$f_2$\n$GA$", "$f_2$\n$SA$"]
    best = []
    scores_list = [a[1] for a in batch_list]
    for scores in scores_list:
        best.append(np.min(scores))
    index = np.arange(6)
    plt.scatter(index, best, c="red", marker='*', s=40)
    plt.ylim(bottom=0)
    plt.xlabel("Metodo di perturbazione ed algoritmo utilizzato")
    plt.xticks(index, labels)
    plt.ylabel("Performance $[$RMSD value$]$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()
    return best


def plot_2D_comparison(batch_case, network_tupla,
                       title, savepath, method="f0"):
    best_index = batch_case[1].index(min(batch_case[1]))
    weight_list = np.asarray([a for a in batch_case[0]])
    best_weights = weight_list[best_index]
    original_coords = network_tupla[1]
    base_coords = nt.get_spectral_coordinates(
        nx.laplacian_matrix(network_tupla[0]).todense(),
        dim=2)
    if method == "f0":
        after_coords = nt.get_spectral_coordinates(
            nx.laplacian_matrix(network_tupla[0]).todense(),
            mod_matrix=nt.create_inverse_mod_matrix(best_weights),
            dim=2)
    elif method == "f1":
        after_coords = nt.get_spectral_coordinates(
            laplacian=nt.create_customized_laplacian(network_tupla[0],
                                                  best_weights),
            dim=2)
    elif method == "f2":
        after_coords = nt.get_spectral_coordinates(
            laplacian=nt.create_weighted_laplacian(network_tupla[0],
                                                   best_weights),
            dim=2)
    else:
        assert False
    base_coords = pd.DataFrame(
        rmsd.kabsch_rotate(base_coords.values,
                           original_coords.values),
        columns=["x", "y", "z"])
    after_coords = pd.DataFrame(
        rmsd.kabsch_rotate(after_coords.values,
                           original_coords.values),
        columns=["x", "y", "z"])
    plt.scatter(original_coords["x"], original_coords["y"],
                color='C0', marker='.', label="Originali", s=50)
    plt.scatter(base_coords["x"], base_coords["y"],
                color='C2', marker='x', label="SD di base", s=10)
    plt.scatter(after_coords["x"], after_coords["y"], s=10,
                color='C1', marker='*', label="SD con modifica")
    plt.title("")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()


def plot_3D_comparison(batch_case, network_tupla,
                       title, savepath, method="f0", angle=30):
    best_index = batch_case[1].index(min(batch_case[1]))
    weight_list = np.asarray([a for a in batch_case[0]])
    best_weights = weight_list[best_index]
    original_coords = network_tupla[1]
    base_coords = nt.get_spectral_coordinates(
        nx.laplacian_matrix(network_tupla[0]).todense(),
        dim=3)
    if method == "f0":
        after_coords = nt.get_spectral_coordinates(
            nx.laplacian_matrix(network_tupla[0]).todense(),
            mod_matrix=nt.create_inverse_mod_matrix(best_weights),
            dim=3)
    elif method == "f1":
        after_coords = nt.get_spectral_coordinates(
            laplacian=nt.create_customized_laplacian(network_tupla[0],
                                                     best_weights),
            dim=3)
    elif method == "f2":
        after_coords = nt.get_spectral_coordinates(
            laplacian=nt.create_weighted_laplacian(network_tupla[0],
                                                   best_weights),
            dim=3)
    else:
        assert False
    base_coords = pd.DataFrame(
        rmsd.kabsch_rotate(base_coords.values,
                           original_coords.values),
        columns=["x", "y", "z"])
    after_coords = pd.DataFrame(
        rmsd.kabsch_rotate(after_coords.values,
                           original_coords.values),
        columns=["x", "y", "z"])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(original_coords["x"], original_coords["y"], original_coords["z"],
                color='C0', marker='.', label="Originali")
    ax.scatter(base_coords["x"], base_coords["y"], base_coords["z"],
                color='C2', marker='x', label="SD base")
    ax.scatter(after_coords["x"], after_coords["y"], after_coords["z"],
                color='C1', marker='*', label="SD con modifica")
    ax.set_title("")
    ax.legend()
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_zlabel("$Z$")
    ax.view_init(30, angle)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()


def plot_1D_masses(batch_case, title, savepath, is_weight=False):
    best_index = batch_case[1].index(min(batch_case[1]))
    weight_list = np.asarray([a for a in batch_case[0]])
    average = np.average(weight_list, axis=0)
    sigma = np.std(weight_list, axis=0)
    index = np.arange(len(average))
    plt.plot(index, weight_list[best_index], marker='o',
             markersize=2, label="Miglior caso")
    plt.errorbar(index, average, yerr=sigma, markersize=1.5,
                 elinewidth=1, marker='o', capsize=3, label="Media")
    if is_weight:
        plt.xlabel("\\# link")
    else:
        plt.xlabel("\\# nodo")
    plt.ylim(bottom=0)
    if is_weight:
        plt.ylabel("Peso $[a.u.]$")
    else:
        plt.ylabel("Massa $[a.u.]$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()
    return average, sigma


def plot_2D_masses(batch_case, title, savepath, use_average=False):
    best_index = batch_case[1].index(min(batch_case[1]))
    weight_list = np.asarray([a for a in batch_case[0]])
    best_weights = weight_list[best_index]
    average = np.average(weight_list, axis=0)
    sigma = np.std(weight_list, axis=0)
    index = np.arange(len(average))
    N = int(np.sqrt(len(average)))
    if use_average:
        matrix = average.reshape((N, N))
    else:
        matrix = best_weights.reshape((N, N))
    plt.imshow(matrix, origin="lower", cmap="viridis",
               norm=matplotlib.colors.Normalize(0,100))
    plt.xlabel("Indice $X$")
    plt.xlabel("Indice $Y$")
    plt.xticks(np.arange(0, N), fontsize=9)
    plt.yticks(np.arange(0, N), fontsize=9)
    plt.title(title)
    c_bar = plt.colorbar()
    c_bar.set_label("Massa $[a.u.]$")
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()
    return average, sigma


def plot_2D_weights(batch_case, std_network, title, savepath,
                    use_average=False):
    best_index = batch_case[1].index(min(batch_case[1]))
    weight_list = np.asarray([a for a in batch_case[0]])
    best_weights = weight_list[best_index]
    average = np.average(weight_list, axis=0)
    sigma = np.std(weight_list, axis=0)
    fig, (ax, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [15, 1]})
    # I hate myself for this...
    N = int(np.sqrt(len(list(std_network.nodes))))
    # Nodes
    nodes = list(itertools.product(range(N), range(N)))
    ax.scatter([a[0] for a in nodes], [a[1] for a in nodes], c="black")
    # Edges...
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    cmap = matplotlib.cm.get_cmap("viridis")
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    for i, j in enumerate(std_network.edges):
        x = np.array([j[0][0], j[1][0]])
        y = np.array([j[0][1], j[1][1]])
        if use_average:
            color = m.to_rgba(average[i])
        else:
            color = m.to_rgba(best_weights[i])
        ax.plot(x, y, c=color, linewidth=2)
    ax.set_xlabel("Indice $X$")
    ax.set_ylabel("Indice $Y$")
    ax.set_xticks(np.arange(0, N))
    ax.set_yticks(np.arange(0, N))
    c_bar = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                             norm=norm,
                                             orientation='vertical')
    c_bar.set_label("Peso $[a.u.]$")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()
    return average, sigma


def plot_3D_masses(batch_case, std_network, cube_size, title, savepath,
                   angle=30):
    # Horrible mess here oh my god...
    weight_list = np.array(batch_case[0])[0]
    node_list = list(itertools.product(range(cube_size),
                                       range(cube_size),
                                       range(cube_size)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, j in enumerate(std_network.edges):
        x = np.array((j[0][0], j[1][0]))
        y = np.array((j[0][1], j[1][1]))
        z = np.array((j[0][2], j[1][2]))
        ax.plot(x, y, z, c="grey", alpha=0.6, linewidth=0.7)
    dots = ax.scatter([a[0] for a in node_list],
                      [a[1] for a in node_list], 
                      [a[2] for a in node_list],
                      c=weight_list, norm=matplotlib.colors.Normalize(0, 100),
                      cmap="viridis")
    c_bar = fig.colorbar(dots)
    c_bar.set_label("Massa $[a.u.]$")
    ax.set_title(title)
    ax.set_xlabel("Indice $X$")
    ax.set_ylabel("Indice $Y$")
    ax.set_zlabel("Indice $Z$")
    ax.set_xticks(np.arange(0, cube_size))
    ax.set_yticks(np.arange(0, cube_size))
    ax.set_zticks(np.arange(0, cube_size))
    ax.view_init(30, angle)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()
    return weight_list


def plot_3D_edges(batch_case, std_network, cube_size, title, savepath,
                  angle=30):
    weight_list = np.array(batch_case[0])[0]
    node_list= list(itertools.product(range(cube_size),
                                      range(cube_size),
                                      range(cube_size)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Nodes
    ax.scatter([a[0] for a in node_list],
                      [a[1] for a in node_list],
                      [a[2] for a in node_list],
                      c="grey")
    # Edges...
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.get_cmap("viridis")
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    for i, j in enumerate(std_network.edges):
        x = np.array((j[0][0], j[1][0]))
        y = np.array((j[0][1], j[1][1]))
        z = np.array((j[0][2], j[1][2]))
        color = m.to_rgba(weight_list[i])
        ax.plot(x, y, z, c=color)
    ax.view_init(30, angle)
    ax.set_xlabel("Indice $X$")
    ax.set_ylabel("Indice $Y$")
    ax.set_zlabel("Indice $Z$")
    ax.set_xticks(np.arange(0, cube_size))
    ax.set_yticks(np.arange(0, cube_size))
    ax.set_zticks(np.arange(0, cube_size))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.clf()
    return weight_list


def rotational_movie(batch_case, std_network, cube_size, title, filename,
                     plot_function, n_frames=360, method=""):
    os.system("mkdir foo")
    os.system("del \"foo\\foo*.jpg\"")
    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        print(str(i) + "/" + str(n_frames))
        if method == "":    
            plot_function(batch_case, std_network, cube_size, title,
                          ("foo\\foo" + str(i).zfill(5) + ".jpg"), angle)
        else:
            plot_function(batch_case, std_network, title,
                          ("foo\\foo" + str(i).zfill(5) + ".jpg"), method,
                          angle)
    os.system("ffmpeg -y -i \"foo\\foo%05d.jpg\" " 
              + "img\\" + filename + ".mp4")


#%%
# Line 10

with open("batch/line_10_f0_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_10_f0_ga = (obtained_weights, scores)

with open("batch/line_10_f1_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_10_f1_ga = (obtained_weights, scores)

with open("batch/line_10_f2_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_10_f2_ga = (obtained_weights, scores)

with open("batch/line_10_f0_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_10_f0_sa = (obtained_weights, scores)

with open("batch/line_10_f1_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_10_f1_sa = (obtained_weights, scores)

with open("batch/line_10_f2_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_10_f2_sa = (obtained_weights, scores)

#%%
print_score_block([line_10_f0_ga, line_10_f0_sa, line_10_f1_ga,
                   line_10_f1_sa, line_10_f2_ga, line_10_f2_sa],
                  "", "img/line_10.jpg")

plot_1D_masses(line_10_f0_ga, "", "img/line_10_f0_ga.jpg")
plot_1D_masses(line_10_f0_sa, "", "img/line_10_f0_sa.jpg")
plot_1D_masses(line_10_f1_ga, "", "img/line_10_f1_ga.jpg")
plot_1D_masses(line_10_f1_sa, "", "img/line_10_f1_sa.jpg")
plot_1D_masses(line_10_f2_ga, "", "img/line_10_f2_ga.jpg", True)
plot_1D_masses(line_10_f2_sa, "", "img/line_10_f2_sa.jpg", True)


#%%
# Line 25

with open("batch/line_25_f0_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_25_f0_ga = (obtained_weights, scores)

with open("batch/line_25_f1_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_25_f1_ga = (obtained_weights, scores)

with open("batch/line_25_f2_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_25_f2_ga = (obtained_weights, scores)

with open("batch/line_25_f0_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_25_f0_sa = (obtained_weights, scores)

with open("batch/line_25_f1_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_25_f1_sa = (obtained_weights, scores)

with open("batch/line_25_f2_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
line_25_f2_sa = (obtained_weights, scores)

#%%

print_score_block([line_25_f0_ga, line_25_f0_sa, line_25_f1_ga,
                   line_25_f1_sa, line_25_f2_ga, line_25_f2_sa],
                  "", "img/line_25.jpg")

plot_1D_masses(line_25_f0_ga, "", "img/line_25_f0_ga.jpg")
plot_1D_masses(line_25_f0_sa, "", "img/line_25_f0_sa.jpg")
plot_1D_masses(line_25_f1_ga, "", "img/line_25_f1_ga.jpg")
plot_1D_masses(line_25_f1_sa, "", "img/line_25_f1_sa.jpg")
plot_1D_masses(line_25_f2_ga, "", "img/line_25_f2_ga.jpg", True)
plot_1D_masses(line_25_f2_sa, "", "img/line_25_f2_sa.jpg", True)

#%%
# Lattice 5

with open("batch/lattice_5_f0_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_5_f0_ga = (obtained_weights, scores)

with open("batch/lattice_5_f1_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_5_f1_ga = (obtained_weights, scores)

with open("batch/lattice_5_f2_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_5_f2_ga = (obtained_weights, scores)

with open("batch/lattice_5_f0_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_5_f0_sa = (obtained_weights, scores)

with open("batch/lattice_5_f1_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_5_f1_sa = (obtained_weights, scores)

with open("batch/lattice_5_f2_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_5_f2_sa = (obtained_weights, scores)

#%%

print_score_block([lattice_5_f0_ga, lattice_5_f0_sa, lattice_5_f1_ga,
                   lattice_5_f1_sa, lattice_5_f2_ga, lattice_5_f2_sa],
                  "", "img/lattice_5.jpg")

plot_2D_masses(lattice_5_f0_ga, "", "img/lattice_5_f0_ga.jpg")
plot_2D_masses(lattice_5_f0_ga, "", "img/lattice_5_f0_ga_average.jpg", True)
plot_2D_masses(lattice_5_f1_ga, "", "img/lattice_5_f1_ga.jpg")
plot_2D_masses(lattice_5_f1_ga, "", "img/lattice_5_f1_ga_average.jpg", True)
plot_2D_masses(lattice_5_f0_sa, "", "img/lattice_5_f0_sa.jpg")
plot_2D_masses(lattice_5_f0_sa, "", "img/lattice_5_f0_sa_average.jpg", True)
plot_2D_masses(lattice_5_f1_sa, "", "img/lattice_5_f1_sa.jpg")
plot_2D_masses(lattice_5_f1_sa, "", "img/lattice_5_f1_sa_average.jpg", True)

#%%
G = nt.create_lattice(5, 5)[0]
plot_2D_weights(lattice_5_f2_ga, G, "", "img/lattice_5_f2_ga.jpg")
plot_2D_weights(lattice_5_f2_sa, G, "", "img/lattice_5_f2_sa.jpg")
plot_2D_weights(lattice_5_f2_ga, G, "",
                "img/lattice_5_f2_ga_average.jpg", True)
plot_2D_weights(lattice_5_f2_sa, G, "",
                "img/lattice_5_f2_sa_average.jpg", True)

#%%
G = nt.create_lattice(5, 5)
plot_2D_comparison(lattice_5_f1_ga, G, "",
                   "img/lattice_5_f1_ga_comparison.jpg", "f1")
plot_2D_comparison(lattice_5_f1_sa, G, "",
                   "img/lattice_5_f1_sa_comparison.jpg", "f1")
plot_2D_comparison(lattice_5_f2_ga, G, "",
                   "img/lattice_5_f2_ga_comparison.jpg", "f2")
plot_2D_comparison(lattice_5_f2_sa, G, "",
                   "img/lattice_5_f2_sa_comparison.jpg", "f2")
plot_2D_comparison(lattice_5_f0_ga, G, "",
                   "img/lattice_5_f0_ga_comparison.jpg", "f0")
plot_2D_comparison(lattice_5_f0_sa, G, "",
                   "img/lattice_5_f0_sa_comparison.jpg", "f0")

#%%
# Lattice 10

with open("batch/lattice_10_f0_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_10_f0_ga = (obtained_weights, scores)

with open("batch/lattice_10_f1_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_10_f1_ga = (obtained_weights, scores)

with open("batch/lattice_10_f2_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_10_f2_ga = (obtained_weights, scores)

with open("batch/lattice_10_f0_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_10_f0_sa = (obtained_weights, scores)

with open("batch/lattice_10_f1_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_10_f1_sa = (obtained_weights, scores)

with open("batch/lattice_10_f2_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_10_f2_sa = (obtained_weights, scores)

#%%

print_score_block([lattice_10_f0_ga, lattice_10_f0_sa, lattice_10_f1_ga,
                   lattice_10_f1_sa, lattice_10_f2_ga, lattice_10_f2_sa],
                  "", "img/lattice_10.jpg")
G = nt.create_lattice(10, 10)[0]

plot_2D_masses(lattice_10_f0_ga, "", "img/lattice_10_f0_ga.jpg")
plot_2D_masses(lattice_10_f0_ga, "", "img/lattice_10_f0_ga_average.jpg", True)
plot_2D_masses(lattice_10_f1_ga, "", "img/lattice_10_f1_ga.jpg")
plot_2D_masses(lattice_10_f1_ga, "", "img/lattice_10_f1_ga_average.jpg", True)
plot_2D_masses(lattice_10_f0_sa, "", "img/lattice_10_f0_sa.jpg")
plot_2D_masses(lattice_10_f0_sa, "", "img/lattice_10_f0_sa_average.jpg", True)
plot_2D_masses(lattice_10_f1_sa, "", "img/lattice_10_f1_sa.jpg")
plot_2D_masses(lattice_10_f1_sa, "", "img/lattice_10_f1_sa_average.jpg", True)

#%%
G = nt.create_lattice(10, 10)[0]
plot_2D_weights(lattice_10_f2_ga, G, "", "img/lattice_10_f2_ga.jpg")
plot_2D_weights(lattice_10_f2_sa, G, "", "img/lattice_10_f2_sa.jpg")
plot_2D_weights(lattice_10_f2_ga, G, "",
                "img/lattice_10_f2_ga_average.jpg", True)
plot_2D_weights(lattice_10_f2_sa, G, "",
                "img/lattice_10_f2_sa_average.jpg", True)

#%%
G = nt.create_lattice(10, 10)
plot_2D_comparison(lattice_10_f1_ga, G, "",
                   "img/lattice_10_f1_ga_comparison.jpg", "f1")
plot_2D_comparison(lattice_10_f1_sa, G, "",
                   "img/lattice_10_f1_sa_comparison.jpg", "f1")
plot_2D_comparison(lattice_10_f2_ga, G, "",
                   "img/lattice_10_f2_ga_comparison.jpg", "f2")
plot_2D_comparison(lattice_10_f2_sa, G, "",
                   "img/lattice_10_f2_sa_comparison.jpg", "f2")
plot_2D_comparison(lattice_10_f0_ga, G, "",
                   "img/lattice_10_f0_ga_comparison.jpg", "f0")
plot_2D_comparison(lattice_10_f0_sa, G, "",
                   "img/lattice_10_f0_sa_comparison.jpg", "f0")

#%%
# Lattice 20

with open("batch/lattice_20_f0_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_20_f0_ga = (obtained_weights, scores)

with open("batch/lattice_20_f1_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_20_f1_ga = (obtained_weights, scores)

with open("batch/lattice_20_f2_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_20_f2_ga = (obtained_weights, scores)

with open("batch/lattice_20_f0_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_20_f0_sa = (obtained_weights, scores)

with open("batch/lattice_20_f1_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_20_f1_sa = (obtained_weights, scores)

with open("batch/lattice_20_f2_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
lattice_20_f2_sa = (obtained_weights, scores)

#%%

print_score_single([lattice_20_f0_ga, lattice_20_f0_sa, lattice_20_f1_ga,
                   lattice_20_f1_sa, lattice_20_f2_ga, lattice_20_f2_sa], "", "img/lattice_20.jpg")

plot_2D_masses(lattice_20_f0_ga, "", "img/lattice_20_f0_ga.jpg")
plot_2D_masses(lattice_20_f1_ga, "", "img/lattice_20_f1_ga.jpg")
plot_2D_masses(lattice_20_f0_sa, "", "img/lattice_20_f0_sa.jpg")
plot_2D_masses(lattice_20_f1_sa, "", "img/lattice_20_f1_sa.jpg")

#%%
G = nt.create_lattice(20, 20)[0]
plot_2D_weights(lattice_20_f2_ga, G, "", "img/lattice_20_f2_ga.jpg")
plot_2D_weights(lattice_20_f2_sa, G, "", "img/lattice_20_f2_sa.jpg")

#%%
G = nt.create_lattice(20, 20)
plot_2D_comparison(lattice_20_f1_ga, G, "",
                   "img/lattice_20_f1_ga_comparison.jpg", "f1")
plot_2D_comparison(lattice_20_f1_sa, G, "",
                   "img/lattice_20_f1_sa_comparison.jpg", "f1")
plot_2D_comparison(lattice_20_f2_ga, G, "",
                   "img/lattice_20_f2_ga_comparison.jpg", "f2")
plot_2D_comparison(lattice_20_f2_sa, G, "",
                   "img/lattice_20_f2_sa_comparison.jpg", "f2")
plot_2D_comparison(lattice_20_f0_ga, G, "",
                   "img/lattice_20_f0_ga_comparison.jpg", "f0")
plot_2D_comparison(lattice_20_f0_sa, G, "",
                   "img/lattice_20_f0_sa_comparison.jpg", "f0")


#%%
# Cube 5

with open("batch/cube_5_f0_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_5_f0_ga = (obtained_weights, scores)

with open("batch/cube_5_f1_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_5_f1_ga = (obtained_weights, scores)

with open("batch/cube_5_f2_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_5_f2_ga = (obtained_weights, scores)

with open("batch/cube_5_f0_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_5_f0_sa = (obtained_weights, scores)

with open("batch/cube_5_f1_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_5_f1_sa = (obtained_weights, scores)

with open("batch/cube_5_f2_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_5_f2_sa = (obtained_weights, scores)

#%%

print_score_single([cube_5_f0_ga, cube_5_f0_sa, cube_5_f1_ga,
                   cube_5_f1_sa, cube_5_f2_ga, cube_5_f2_sa],
                  "", "img/cube_5.jpg")

#%%
G = nt.create_cube(5, 5, 5)[0]

plot_3D_masses(cube_5_f0_ga, G, 5, "", "img/cube_5_f0_ga.jpg")
plot_3D_masses(cube_5_f0_sa, G, 5, "", "img/cube_5_f0_sa.jpg")
plot_3D_masses(cube_5_f1_ga, G, 5, "", "img/cube_5_f1_ga.jpg")
plot_3D_masses(cube_5_f1_sa, G, 5, "", "img/cube_5_f1_sa.jpg")

#%%
G = nt.create_cube(5, 5, 5)[0]
plot_3D_edges(cube_5_f2_ga, G, 5, "", "img/cube_5_f2_ga.jpg")
plot_3D_edges(cube_5_f2_sa, G, 5, "", "img/cube_5_f2_sa.jpg")

#%%
G = nt.create_cube(5, 5, 5)[0]
G_tupla = nt.create_cube(5, 5, 5)

plot_3D_comparison(cube_5_f0_ga, G_tupla, "", "img/cube_5_f0_ga_comparison.jpg", method="f0")
plot_3D_comparison(cube_5_f1_ga, G_tupla, "", "img/cube_5_f1_ga_comparison.jpg", method="f1")
plot_3D_comparison(cube_5_f2_ga, G_tupla, "", "img/cube_5_f2_ga_comparison.jpg", method="f2")
plot_3D_comparison(cube_5_f0_sa, G_tupla, "", "img/cube_5_f0_sa_comparison.jpg", method="f0")
plot_3D_comparison(cube_5_f1_sa, G_tupla, "", "img/cube_5_f1_sa_comparison.jpg", method="f1")
plot_3D_comparison(cube_5_f2_sa, G_tupla, "", "img/cube_5_f2_sa_comparison.jpg", method="f2")

#%%
G = nt.create_cube(5, 5, 5)[0]
G_tupla = nt.create_cube(5, 5, 5)

rotational_movie(cube_5_f0_ga, G_tupla, 5, "", "cube_5_f0_ga",
                 plot_3D_comparison, 360, "f0")
rotational_movie(cube_5_f0_sa, G_tupla, 5, "", "cube_5_f0_sa",
                 plot_3D_comparison, 360, "f0")
rotational_movie(cube_5_f1_ga, G_tupla, 5, "", "cube_5_f1_ga",
                 plot_3D_comparison, 360, "f1")
rotational_movie(cube_5_f1_sa, G_tupla, 5, "", "cube_5_f1_sa",
                 plot_3D_comparison, 360, "f1")
rotational_movie(cube_5_f2_ga, G_tupla, 5, "", "cube_5_f2_ga",
                 plot_3D_comparison, 360, "f2")
rotational_movie(cube_5_f2_sa, G_tupla, 5, "", "cube_5_f2_sa",
                 plot_3D_comparison, 360, "f2")

rotational_movie(cube_5_f0_ga, G, 5, "", "cube_5_f0_ga_weights",
                 plot_3D_masses, 360)
rotational_movie(cube_5_f0_sa, G, 5, "", "cube_5_f0_sa_weights",
                 plot_3D_masses, 360)
rotational_movie(cube_5_f1_ga, G, 5, "", "cube_5_f1_ga_weights",
                 plot_3D_masses, 360)
rotational_movie(cube_5_f1_sa, G, 5, "", "cube_5_f1_sa_weights",
                 plot_3D_masses, 360)
rotational_movie(cube_5_f2_ga, G, 5, "", "cube_5_f2_ga_weights",
                 plot_3D_edges, 360)
rotational_movie(cube_5_f2_sa, G, 5, "", "cube_5_f2_sa_weights",
                 plot_3D_edges, 360)

#%%
# Cube 7

with open("batch/cube_7_f0_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_7_f0_ga = (obtained_weights, scores)

with open("batch/cube_7_f1_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_7_f1_ga = (obtained_weights, scores)

with open("batch/cube_7_f2_ga.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_7_f2_ga = (obtained_weights, scores)

with open("batch/cube_7_f0_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_7_f0_sa = (obtained_weights, scores)

with open("batch/cube_7_f1_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_7_f1_sa = (obtained_weights, scores)

with open("batch/cube_7_f2_sa.pkl", 'rb') as f:
    obtained_weights, scores = pickle.load(f)
cube_7_f2_sa = (obtained_weights, scores)

#%%

print_score_single([cube_7_f0_ga, cube_7_f0_sa, cube_7_f1_ga,
                   cube_7_f1_sa, cube_7_f2_ga, cube_7_f2_sa],
                  "", "img/cube_7.jpg")
#%%
G = nt.create_cube(7, 7, 7)[0]

plot_3D_masses(cube_7_f0_ga, G, 7, "", "img/cube_7_f0_ga.jpg")
plot_3D_masses(cube_7_f0_sa, G, 7, "", "img/cube_7_f0_sa.jpg")
plot_3D_masses(cube_7_f1_ga, G, 7, "", "img/cube_7_f1_ga.jpg")
plot_3D_masses(cube_7_f1_sa, G, 7, "", "img/cube_7_f1_sa.jpg")

#%%
G = nt.create_cube(7, 7, 7)[0]
plot_3D_edges(cube_7_f2_ga, G, 7, "", "img/cube_7_f2_ga.jpg")
plot_3D_edges(cube_7_f2_sa, G, 7, "", "img/cube_7_f2_sa.jpg")

#%%
G = nt.create_cube(7, 7, 7)[0]
G_tupla = nt.create_cube(7, 7, 7)

plot_3D_comparison(cube_7_f0_ga, G_tupla, "", "img/cube_7_f0_ga_comparison.jpg", method="f0")
plot_3D_comparison(cube_7_f1_ga, G_tupla, "", "img/cube_7_f1_ga_comparison.jpg", method="f1")
plot_3D_comparison(cube_7_f2_ga, G_tupla, "", "img/cube_7_f2_ga_comparison.jpg", method="f2")
plot_3D_comparison(cube_7_f0_sa, G_tupla, "", "img/cube_7_f0_sa_comparison.jpg", method="f0")
plot_3D_comparison(cube_7_f1_sa, G_tupla, "", "img/cube_7_f1_sa_comparison.jpg", method="f1")
plot_3D_comparison(cube_7_f2_sa, G_tupla, "", "img/cube_7_f2_sa_comparison.jpg", method="f2")

#%%
G = nt.create_cube(7, 7, 7)[0]
G_tupla = nt.create_cube(7, 7, 7)

rotational_movie(cube_7_f0_ga, G_tupla, 7, "", "cube_7_f0_ga",
                 plot_3D_comparison, 360, "f0")
rotational_movie(cube_7_f0_sa, G_tupla, 7, "", "cube_7_f0_sa",
                 plot_3D_comparison, 360, "f0")
rotational_movie(cube_7_f1_ga, G_tupla, 7, "", "cube_7_f1_ga",
                 plot_3D_comparison, 360, "f1")
rotational_movie(cube_7_f1_sa, G_tupla, 7, "", "cube_7_f1_sa",
                 plot_3D_comparison, 360, "f1")
rotational_movie(cube_7_f2_ga, G_tupla, 7, "", "cube_7_f2_ga",
                 plot_3D_comparison, 360, "f2")
rotational_movie(cube_7_f2_sa, G_tupla, 7, "", "cube_7_f2_sa",
                 plot_3D_comparison, 360, "f2")

rotational_movie(cube_7_f0_ga, G, 7, "", "cube_7_f0_ga_weights",
                 plot_3D_masses, 360)
rotational_movie(cube_7_f0_sa, G, 7, "", "cube_7_f0_sa_weights",
                 plot_3D_masses, 360)
rotational_movie(cube_7_f1_ga, G, 7, "", "cube_7_f1_ga_weights",
                 plot_3D_masses, 360)
rotational_movie(cube_7_f1_sa, G, 7, "", "cube_7_f1_sa_weights",
                 plot_3D_masses, 360)
rotational_movie(cube_7_f2_ga, G, 7, "", "cube_7_f2_ga_weights",
                 plot_3D_edges, 360)
rotational_movie(cube_7_f2_sa, G, 7, "", "cube_7_f2_sa_weights",
                 plot_3D_edges, 360)

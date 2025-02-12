import numpy as np
import pandas as pd
from src.train_load_composition import prepare_input, distribute_load, append_loads
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":

    depth = 6.
    distribution_angle = 45.

    with open('data/trains/data_trains.json', 'r') as f:
        data_all_trains = json.load(f)

    keys = list(data_all_trains.keys())
    key = keys[0]

    data_train = data_all_trains[key]
    data_train = {key: np.asarray(val) for (key, val) in data_train.items()}


    """ Analysis for specific sensor depth and angle. """
    load_times, loads, velocity, axle_distances, pwp_times, pressures = prepare_input(data_train)
    times, loads = append_loads(load_times, loads, pwp_times)
    distributed_loads = distribute_load(times, loads, velocity, axle_distances, depth, load_times.min(), load_times.max(), distribution_angle)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    ax.plot(times, loads, c="b")
    ax2.plot(times, distributed_loads, c="r")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Axle load [kN]", fontsize=12)
    ax2.set_ylabel("Stress distributed\nat {d:.1f}m [kPa]".format(d=depth), fontsize=12)
    ax.grid()
    plt.close()
    fig.savefig(r"results/trains/train_distributed_load_timelines.png")


    """ Sensitivity analysis for the distribution angle. """
    depth = 6.
    distribution_angles = np.arange(15, 90, 15)
    distributed_loads = np.zeros((distribution_angles.size, loads.size))
    for i_angle, distribution_angle in enumerate(distribution_angles):
        distributed_loads[i_angle] = distribute_load(times, loads, velocity, axle_distances, depth, load_times.min(), load_times.max(), distribution_angle)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, loads, label="Load [kN]")
    for i_angle, distributed_load in enumerate(distributed_loads):
        ax.plot(times, distributed_load, label="{a:.1f}".format(a=distribution_angles[i_angle]))
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Stress distributed\nat {d:.1f}m [kPa]".format(d=depth), fontsize=12)
    ax.grid()
    ax.legend(title="Distribution angle", fontsize=10, loc="right")
    plt.close()
    fig.savefig(r"results/trains/train_distributed_load_angles.png")


    """ Sensitivity analysis for the sensor depth. """
    distribution_angle = 45
    depths = np.arange(3.6, 6, 0.8)
    distributed_loads = np.zeros((depths.size, loads.size))
    for i_depth, depth in enumerate(depths):
        distributed_loads[i_depth] = distribute_load(times, loads, velocity, axle_distances, depth, load_times.min(), load_times.max(), distribution_angle)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, loads, label="Load [kN]")
    for i_depth, distributed_load in enumerate(distributed_loads):
        ax.plot(times, distributed_load, label="{a:.1f}".format(a=depths[i_depth]))
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Distributed stress with angle {a:0f} [kPa]".format(a=distribution_angle).format(d=depth), fontsize=12)
    ax.grid()
    ax.legend(title="Depth", fontsize=10, loc="right")
    plt.close()
    fig.savefig(r"results/trains/train_distributed_load_depths.png")


    # pp = PdfPages(r"results/trains/train_distributed_load_timelines.pdf")
    # [pp.savefig(fig) for fig in figs]
    # pp.close()

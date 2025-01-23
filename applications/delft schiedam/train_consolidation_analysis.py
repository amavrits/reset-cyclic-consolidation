import numpy as np
import pandas as pd
from src.train_load_composition import prepare_input, distribute_load, append_loads, interpolate_pwp
from src.consolidation import dynamic_consolidation_fourrier
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":

    depth = 6
    distribution_angle = 75
    cv = 1e-0
    H = 7
    z_grid = np.asarray([0, H])

    with open('data/trains/data_trains.json', 'r') as f:
        data_all_trains = json.load(f)

    keys = list(data_all_trains.keys())
    key = keys[0]

    data_train = data_all_trains[key]
    data_train = {key: np.asarray(val) for (key, val) in data_train.items()}

    load_times, loads, velocity, axle_distances, pwp_times, pressures = prepare_input(data_train)
    times, loads = append_loads(load_times, loads, pwp_times)
    distributed_loads = distribute_load(times, loads, velocity, axle_distances, depth, load_times.min(), load_times.max(), distribution_angle)

    pressures_interp = interpolate_pwp(times, pwp_times, pressures)
    pressures -= pressures[0]



    idx_start = np.argwhere(times>40).min()
    idx_end = np.argwhere(times<100).max()
    times = times[idx_start:idx_end]
    distributed_loads = distributed_loads[idx_start:idx_end]
    distributed_loads = np.where(distributed_loads>0,distributed_loads.mean(), 0)
    from src.fourrier import decompose
    A, B, angular_freqs, y = decompose(distributed_loads, times, False)
    u_model = dynamic_consolidation_fourrier(distributed_loads, cv, z_grid, times).squeeze()

    fig = plt.figure()
    plt.plot(times, distributed_loads, c="k", label="Total stress")
    plt.plot(times, y, c="g", label="Total stress")
    # plt.plot(times, pressures_interp, c="b", label="Overpressure data")
    plt.plot(pwp_times, pressures, c="b", label="Overpressure data")
    plt.plot(times, u_model[0], c="r", label="Overpressure model")
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Stress or overpressure [kPa]", fontsize=12)
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True)
    plt.grid()
    plt.close()
    fig.savefig(r"results/trains/pressure_timelines.png")

    # fig = plt.figure()
    # for i in range(0, time_grid.size, 10):
    #     t = time_grid[i]
    #     plt.plot(u[:, i], z_grid / H, label=str(round(t, 1)))
    # plt.xlabel("Overpressure [kPa]", fontsize=12)
    # plt.ylabel("z/H [-]", fontsize=12)
    # plt.legend(title="Time [s]")
    # plt.grid()
    # plt.close()
    # fig.savefig(r"results/pressure_height.png")

    # pp = PdfPages(r"results/trains/train_pressure_timelines.pdf")
    # [pp.savefig(fig) for fig in figs]
    # pp.close()


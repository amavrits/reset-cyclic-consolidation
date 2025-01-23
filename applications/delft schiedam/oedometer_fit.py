import os

import jaxopt
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from jaxopt import GradientDescent
from src.consolidation import dynamic_consolidation, jax_dynamic_consolidation, jax_oedometer
import matplotlib.pyplot as plt


if __name__ == "__main__":

    crs_folder = r"data/crs tests"
    crs_files = os.listdir(crs_folder)

    crs_data = {}
    for crs_file in crs_files[1:]:
        path = os.path.join(crs_folder, crs_file)
        with open(path, "r") as f:
            data_lines = f.readlines()
        n_columns = int(data_lines[4].split("= ")[-1].strip("\n"))
        columns = [x.split(",")[2].strip(" ") for x in data_lines[5:5+n_columns]]
        # TODO:
        init_height = 20 / 1_000
        diameter = 63 / 1_000
        load = 2
        start_line = data_lines.index("#EOH=\n")
        data_lines = data_lines[start_line+1:]
        for i, line in enumerate(data_lines):
            line = line.strip("\n")
            line = line.split(" ")
            line = [float(x) for x in line if x != ""]
            data_lines[i] = line
        data = pd.DataFrame(data=np.r_[data_lines], columns=columns)
        data = data.rename(columns={"Tijd": "time", "Verplaatsing": "s", "Poriendruk": "u", "tijd": "time",
                                    "zakking": "s", "spanning": "sigma"})
        data["s"] /= 1_000  # Convert mm to m
        crs_data[crs_file] = {"data": data, "init_height": init_height, "diameter": diameter, "load": load}

    data, h, d, load = tuple(crs_data[crs_files[-1]]. values())
    area = np.pi * d ** 2 / 4
    stress = load / area
    # time = data["time"].values[:10310]  # Fit start of timeline
    # sigma_eff = data["sigma"].values[:10310]  # Fit start of timeline
    time = data["time"].values[29781:29781+2290]  # Fit reload of timeline
    sigma_eff = data["sigma"].values[29781:29781+2290]  # Fit reload of timeline
    u_data = stress - sigma_eff
    z = np.asarray([h])

    fig = plt.figure()
    plt.scatter(time/3600, u_data)
    plt.close()
    fig.savefig(r"results/crs/oedometer_data.png")

    def f_overpressure(x):
        u = jax_oedometer(x, stress, z, time).squeeze()
        return u

    def f_mse(x):
        u = f_overpressure(x)
        mse = 0.5 * jnp.linalg.norm((u-u_data)**2) / u.size
        return mse

    cvs = np.linspace(1e-10, 1e-9, 100)
    mses = np.asarray(jax.vmap(f_mse)(cvs))
    cv_hat = cvs[np.argmin(mses)]
    mse = f_mse(cv_hat)
    fit = f_overpressure(cv_hat)

    fig = plt.figure()
    plt.plot(cvs, mses)
    plt.scatter([cv_hat], [mse], c="r")
    plt.xlabel("${C}_{v}$ [${m}^{2}$/s]", fontsize=12)
    plt.ylabel("MSE [-]", fontsize=12)
    plt.close()
    fig.savefig(r"results/crs/oedometer_fit_mse.png")

    fig = plt.figure()
    plt.axhline(stress, c="k", label="Load")
    plt.scatter(time/3_600, u_data, c="b", label="Data")
    plt.plot(time/3_600, fit, c="r", label="Fit")
    plt.xlabel("Time [hr]", fontsize=12)
    plt.ylabel("Overpressure [kPa]", fontsize=12)
    plt.legend()
    plt.close()
    fig.savefig(r"results/crs/oedometer_fit.png")

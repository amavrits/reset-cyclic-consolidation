import numpy as np
import pandas as pd


def distribute_load(times, loads, velocity, axle_distances, depth, load_start_time, load_end_time, distribution_angle=45):

    distribution_angle_rad = distribution_angle * np.pi / 180

    distribution_half_base = depth * np.tan(distribution_angle_rad)
    distribution_base = 2 * distribution_half_base
    distribution_area = distribution_base ** 2
    distribution_stresses = loads / distribution_area

    """ Add axle distances for zero loads """
    times_prior_train = times[np.argwhere(times<load_start_time)].squeeze()
    dtimes_prior_train = np.append(0, np.diff(times_prior_train))
    times_post_train = times[np.argwhere(times>load_end_time)].squeeze()
    dtimes_post_train = np.append(np.diff(times_post_train), 0)
    train_prior_distance = dtimes_prior_train * velocity
    train_post_distance = dtimes_post_train * velocity
    axle_distances[0] = (times[np.argwhere(times<load_start_time).max()+1] - times_prior_train[-1]) * velocity
    axle_distances = np.concatenate((train_prior_distance, axle_distances, train_post_distance))
    cum_axle_distances = np.cumsum(axle_distances)

    traveled_distances = velocity * times
    travelled_axle_distances = np.abs(cum_axle_distances[:, None] - traveled_distances[None, :])
    distributed_loads = np.where(travelled_axle_distances <= distribution_base, distribution_stresses[:, None], 0)
    distributed_loads = distributed_loads.sum(axis=1)

    return distributed_loads


def prepare_input(data):

    load_times = data["load_times"]
    loads = data["loads"]
    velocity = data["velocity"]
    axle_distances = data["axle_distances"]
    pwp_times = data["pwp_times"]
    pressures = data["pressures"]

    return load_times, loads, velocity, axle_distances, pwp_times, pressures


def parse_data(df_trains, df_pwp, train_id, pwp_index = 2):

    df_train = df_trains.loc[df_trains["Train_id"] == train_id]

    start_time = df_train['Start_plot'].min()
    end_time = df_train['End_plot'].max()  # Use Dask to find the start and end indices
    start_index = df_pwp[df_pwp["Time"] > start_time].index.min()
    end_index = df_pwp[df_pwp["Time"] < end_time].index.max()
    df_pwp_train = df_pwp.iloc[start_index: end_index]

    pwp_times = pd.to_datetime(list(df_pwp_train.loc[:, "Time"].values))
    pwp_start_time = pwp_times[0]
    pwp_times = (pwp_times - pwp_times[0]).total_seconds().values
    pressures = df_pwp_train.iloc[:, 2].values

    load_times = pd.to_datetime(list(df_train["Time_vertical_lines"]))
    load_times = (load_times - pwp_start_time).total_seconds().values
    loads = df_train["Askwaliteit_aslast_ton"].values * 10  # Tons to kN
    velocity = df_train["Askwaliteit_assnelheid_ms"].values.mean()
    axle_distances = df_train["Askwaliteit_asafstand"].values

    data_train = {
        "load_times": load_times,
        "loads": loads,
        "velocity": velocity,
        "axle_distances": axle_distances,
        "pwp_times": pwp_times,
        "pressures": pressures
    }

    return data_train


def append_loads(load_times, loads, pwp_times):

    load_start_index = np.argwhere(pwp_times < load_times.min()).max()
    load_end_index = np.argwhere(pwp_times > load_times.max()).min()
    load_times_appended = np.concatenate((pwp_times[:load_start_index], load_times, pwp_times[load_end_index:]))
    loads_appended = np.concatenate((np.zeros(load_start_index), loads, np.zeros(pwp_times.size-load_end_index)))

    return load_times_appended, loads_appended


def interpolate_pwp(times, pwp_times, pwp):
    return np.interp(times, pwp_times, pwp)


if __name__ == "__main__":

    pass

import numpy as np
import pandas as pd
from src.train_load_composition import parse_data
import json
from tqdm import tqdm


if __name__ == "__main__":

    all_train_passages = pd.read_csv(r"data/trains/pressures_vs_weight.csv")
    pwp_sensors = pd.read_parquet(r"data/trains/Sensor_data_left_location_75.parquet")

    data_all_trains = {}
    for i, train_id in enumerate(tqdm(pd.unique(all_train_passages["Train_id"]))):

        data_train = parse_data(all_train_passages, pwp_sensors, train_id, pwp_index=2)
        data_train = {key: val.tolist() for (key, val) in data_train.items()}
        data_all_trains[train_id.item()] = data_train

    with open('data/trains/data_trains.json', 'w') as f:
        json.dump(data_all_trains, f)


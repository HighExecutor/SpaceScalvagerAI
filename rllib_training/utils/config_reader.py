import os
import yaml
import pickle


def read_yaml(path_to_yaml) -> dict:
    with open(path_to_yaml) as stream:
        data = yaml.safe_load(stream)
    return data


def read_config(checkpoint_path) -> dict:
    with open(os.path.join(checkpoint_path, "algorithm_state.pkl"), 'rb') as file:
        config = pickle.load(file)
    return config

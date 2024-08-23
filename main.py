from sample_generator import VRPGenerator
from model_arch import TransformerVRPModel
import tensorflow as tf
import numpy as np
from config import Config
from train import Train


def create_ds_set(num_samples, max_num_loads, path):
    vrp_generator = VRPGenerator(num_files=num_samples, max_loads=max_num_loads)
    vrp_generator.create_multiple_vrp_files(directory=path)


def train_model(cnf: Config):
    tr_h = Train(cnf=cnf)
    tr_h.train()


if __name__ == "__main__":
    # create_model()
    cnf = Config()

    tasks = {
        'ds_creation': 1,
        'train': 1
    }
    if tasks['ds_creation'] == 1:
        create_ds_set(num_samples=1000, max_num_loads=cnf.num_loads, path=cnf.ds_train_path)
        create_ds_set(num_samples=10, max_num_loads=cnf.num_loads, path=cnf.ds_test_path)

    if tasks['train'] == 1:
        train_model(cnf=cnf)

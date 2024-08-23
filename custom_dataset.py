import os

import numpy as np

from config import Config, DatasetType
from random import shuffle
import  tensorflow as tf

class CustomDataset:
    def __init__(self, cnf: Config, ds_type:int = DatasetType().TRAIN):
        self.cnf = cnf
        self.ds_type = ds_type

    def create_dataset(self):
        ds_path = self.cnf.ds_train_path
        if self.ds_type == DatasetType().TEST:
            ds_path = self.cnf.ds_test_path

        dataset = self._parse_all_files(ds_path)
        return dataset

    def _parse_all_files(self, ds_path):
        dataset = []

        files_list = os.listdir(ds_path)
        shuffle(files_list)

        for index, file_address in enumerate(files_list):
            dataset.append(self._parse(os.path.join(ds_path, file_address)))

        return dataset

    def iterate(self, dataset, batch_size):
        batch_sequence = []

        for value in dataset:
            batch_sequence.append(value)
            if len(batch_sequence) == batch_size:
                yield tf.cast(batch_sequence, tf.float32)
                batch_sequence = []

        # Yield the remaining elements if they exist
        if batch_sequence:
            yield batch_sequence

    def _parse(self, file_address):
        loads = []
        '''read'''
        with open(file_address, 'r') as file:
            lines = file.readlines()

            for i in range(1, len(lines)):
                load_number, pickup, dropoff = lines[i].strip().split()
                loads.append(np.array([float(load_number), eval(pickup)[0], eval(pickup)[1], eval(dropoff)[0], eval(dropoff)[1]]))
            '''pad '''
            while len(loads) < self.cnf.num_loads:
                loads.append(np.array([-1.0, 0.0,0.0,0.0,0.0]))
        return np.array(loads)

    def pad_tensor(self, input_tensor):
        # Define the maximum number of loads
        max_loads = self.cnf.num_loads

        # Get the shape of the input tensor
        batch_size, num_loads, num_features = tf.shape(input_tensor)

        # Create a tensor of zeros with shape (batch_size, max_loads, num_features)
        padded_tensor = tf.zeros((batch_size, max_loads, num_features), dtype=input_tensor.dtype)

        # Create indices for slicing the input tensor
        indices = tf.slice(input_tensor, [0, 0, 0], [batch_size, tf.minimum(num_loads, max_loads), num_features])

        # Assign the sliced input to the padded tensor
        padded_tensor = tf.tensor_scatter_nd_update(padded_tensor, [[i, j] for i in range(batch_size) for j in
                                                                    range(tf.minimum(num_loads, max_loads))],
                                                    tf.reshape(indices, [-1, num_features]))

        return padded_tensor

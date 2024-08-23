from config import Config
from custom_dataset import CustomDataset
from model_arch import TransformerVRPModel

import tensorflow as tf
import numpy as np
import math


class Train:
    def __init__(self, cnf: Config):
        self.cnf = cnf

    def train(self):
        """"""
        ''' create dataset'''
        ds_h = CustomDataset(self.cnf)  # dataset helper
        dataset = ds_h.create_dataset()

        '''create model'''
        optimizer = tf.keras.optimizers.Adam(self.cnf.lr)
        model = TransformerVRPModel(self.cnf.num_loads,
                                    self.cnf.d_model, self.cnf.num_heads,
                                    self.cnf.ff_dim,
                                    self.cnf.num_layers,
                                    self.cnf.dropout_rate)

        '''train loop'''
        for epoch in range(self.cnf.epoch):
            for b_index, batch_data in enumerate(ds_h.iterate(dataset, self.cnf.batch_size)):
                solutions = self._train_step(epoch, b_index, model, batch_data, optimizer)

                '''print a sample solution after 100 step'''
                if b_index % 100 ==0:
                    print('sample load => ')
                    print(batch_data[0])
                    print('solution => ')
                    print(solutions[0])
                    print('============')


    def _train_step(self, epoch, b_index, model, batch_data, optimizer):
        with tf.GradientTape() as tape:
            output = model(batch_data, training=True)
            '''loss'''
            total_loss, driver_cost, routes_cost, solutions = self._calculate_loss(
                input_data=batch_data,
                output_data=output)
            gradients_of_transformer = tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(
                zip(gradients_of_transformer, model.trainable_variables))
            '''print'''
            tf.print("->EPOCH: ", str(epoch),
                     "->STEP: ", str(b_index),
                     ' -> : total_loss: ', total_loss,
                     ' -> : driver_cost: ', driver_cost,
                     ' -> : routes_cost: ', routes_cost,
                     )
        return solutions

    def _calculate_loss(self, input_data, output_data):
        load_ids = np.array(input_data)[:, :, 0]
        solutions = TransformerVRPModel.post_process(load_ids=load_ids, output=output_data)
        num_drivers = tf.cast([len(solution_batch) for solution_batch in solutions], tf.float32)

        '''policy loss'''
        driver_cost = tf.cast(500.0 * num_drivers, tf.float32)
        routes_cost = self._route_cost(input_data=input_data, solutions=solutions)
        total_cost = routes_cost + driver_cost
        '''log prob of action taken: I use policy gradient loss:
        so, my transformer implicitly provides the policy. Specifically, 
        for each load, it assigns drivers based on the probability. 
        I know this solution can be improved much more, 
        but this could be a baseline.'''

        log_probs = - tf.math.log(output_data + 1e-10)

        total_loss = self.policy_gradient_loss(log_probs, total_cost)

        return total_loss, tf.reduce_mean(driver_cost), tf.reduce_mean(routes_cost), solutions

    def _route_cost(self, input_data, solutions):
        def calculate_single_path_cost(single_path, load_configuration):
            """given a sequence of loads, calculate total cost based on path config"""
            load_configuration = np.array(load_configuration)
            depot = (0.0, 0.0)
            current = depot
            s_cost = 0
            for load_id in single_path:
                next_pick = np.array(load_configuration)[int(load_id) - 1][1:3]  # loads start from 1
                next_drop = np.array(load_configuration)[int(load_id) - 1][3:5]  # loads start from 1

                current_to_next_pick = math.sqrt((current[0] - next_pick[0]) ** 2 + (current[1] - next_pick[1]) ** 2)
                next_pick_to_drop = math.sqrt((next_drop[0] - next_pick[0]) ** 2 + (next_drop[1] - next_pick[1]) ** 2)
                s_cost += current_to_next_pick + next_pick_to_drop
                current = next_drop
            '''back to depot'''
            s_cost += math.sqrt((current[0] - depot[0]) ** 2 + (current[1] - depot[1]) ** 2)
            '''check if load is valid'''
            if s_cost <= self.cnf.max_valid_route:
                '''penalize the the model more'''
                s_cost *= self.cnf.invalid_penalty  # we need to actually increase it as model starts learning
            return s_cost

        '''cost for each batch'''
        cost_lists = []
        for i, solution in enumerate(solutions):
            load_conf = input_data[i]
            batch_cost = 0
            for single_paths in solution:
                batch_cost += calculate_single_path_cost(
                    single_path=single_paths,
                    load_configuration=load_conf
                )
            cost_lists.append(batch_cost)
        return tf.cast(cost_lists, tf.float32)

    def policy_gradient_loss(self, log_probs, cost):
        cost = tf.reshape(cost, (self.cnf.batch_size, 1, 1))
        loss = tf.reduce_mean(log_probs * cost)
        return loss

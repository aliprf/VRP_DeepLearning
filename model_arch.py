import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, Dropout, LayerNormalization
from tensorflow.keras import Model
import numpy as np


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.attention(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class TransformerVRPModel(Model):
    def __init__(self, num_loads, d_model, num_heads, ff_dim, num_layers, dropout_rate=0.1):
        super(TransformerVRPModel, self).__init__()
        self.input_layer = Dense(d_model)  # For input embedding
        self.position_encoding = self.add_weight(
            name="position_encoding",
            shape=[1, num_loads, d_model],
            initializer="zeros",
            trainable=False
        )
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout_rate) for _ in
                               range(num_layers)]
        self.final_layer = Dense(num_loads, activation='softmax')  # To get probability distribution over loads

    # def adjust_input_to_512(self, input_tensor):
    #     # Define the maximum number of loads
    #     max_loads = 512
    #
    #     # Get the shape of the input tensor
    #     batch_size, num_loads, num_features = tf.shape(input_tensor)
    #
    #     # Create a tensor of zeros with shape (batch_size, max_loads, num_features)
    #     padded_tensor = tf.zeros((batch_size, max_loads, num_features), dtype=input_tensor.dtype)
    #
    #     # Create indices for slicing the input tensor
    #     indices = tf.slice(input_tensor, [0, 0, 0], [batch_size, tf.minimum(num_loads, max_loads), num_features])
    #
    #     # Assign the sliced input to the padded tensor
    #     padded_tensor = tf.tensor_scatter_nd_update(padded_tensor, [[i, j] for i in range(batch_size) for j in
    #                                                                 range(tf.minimum(num_loads, max_loads))],
    #                                                 tf.reshape(indices, [-1, num_features]))
    #
    #     return padded_tensor

    def call(self, inputs, training=False, mask=None):
        x = self.input_layer(inputs)  # Apply input embedding
        seq_len = tf.shape(inputs)[1]
        x += self.position_encoding[:, :seq_len, :]  # Add position encoding

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training, mask=mask)

        output = self.final_layer(x)  # (batch_size, seq_len, num_loads)
        return output

    @staticmethod
    def post_process(load_ids, output):
        batch_size, seq_len, num_loads = tf.shape(output)
        load_assignments = []

        for b in range(batch_size):
            # probabilities for each load
            assignment_probs = output[b].numpy()

            vehicle_loads = {}

            probabilities_vehicle_loads= {}

            load_id = load_ids[b]

            for i in range(seq_len):
                if load_id[i] == -1.0:
                    continue
                # Get the load index with the highest probability for the current position
                load_idx = np.argmax(assignment_probs[i])
                # todo: record the probability too,
                prob = assignment_probs[i][load_idx]
                # If this load index is not assigned yet, assign it
                if load_idx not in vehicle_loads:
                    # Create a new list for this vehicle
                    vehicle_loads[load_idx] = []
                    # record probability for sort the sequence of loads
                    probabilities_vehicle_loads[load_idx]= []

                # Add the current load index to the corresponding vehicle
                vehicle_loads[load_idx].append(load_id[i])  # Convert 0-based index to 1-based load ID
                probabilities_vehicle_loads[load_idx].append(prob)
                # print('1')

            list_vehicle_loads = list(vehicle_loads.items())
            list_vehicle_loads_probabilities = list(probabilities_vehicle_loads.items())

            # remove the key
            list_vehicle_loads = [loads for _, loads in list_vehicle_loads]
            list_vehicle_loads_probabilities = [loads for _, loads in list_vehicle_loads_probabilities]

            '''reorder the loads for each vehicle based on the probability of each load'''
            for i in range(len(list_vehicle_loads)):
                list_values = list_vehicle_loads[i]
                list_reference = list_vehicle_loads_probabilities[i]
                revised_values = [item for _, item in
                                  sorted(zip(list_reference, list_values), key=lambda x: x[0], reverse=True)]
                list_vehicle_loads[i] = revised_values

            load_assignments.append(list_vehicle_loads)

        return load_assignments

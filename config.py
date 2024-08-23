class Config:
    def __init__(self):
        """"""
        '''constraint'''
        self.max_valid_route = 720
        self.invalid_penalty = 10
        '''data set'''
        self.ds_train_path: str = f'./ds/train/'
        self.ds_test_path: str = f'./ds/validation/'

        '''transformer config'''
        self.num_loads = 50  # Number of loads
        self.d_model = 64  # Dimension of the model
        self.num_heads = 8  # Number of attention heads
        self.ff_dim = 128  # Feed-forward network dimension
        self.num_layers = 2  # Number of encoder layers
        self.dropout_rate = 0.1  # Dropout rate

        '''train config'''
        self.batch_size = 25
        self.epoch = 10
        self.lr = 1e-3


class DatasetType:
    def __init__(self):
        self.TRAIN: int = 0
        self.TEST: int = 1

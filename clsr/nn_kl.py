from .base import BaseNetwork
from .nn_2l import TwoLayerNeuralNetwork

# class KLayerNeuralNetwork(BaseNetwork):
class KLayerNeuralNetwork(TwoLayerNeuralNetwork):

    DEFAULT_PARAMS = {
        "dtype": "float32",
        "verbose": True,
        "wgt_init": "xavier",
        "p_dropout": 0.0,
        "n_features": 3072,
        "n_classes": 10,
        "n_hidden_nodes": [50],
        "batch_norm": True
    }

    def __init__(self, **params):
        self.set_params(**params)
        if self.verbose:
            self.print_instance_config()
        self.training = True

        # init params
        self.initalize_wgts()
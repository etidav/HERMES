import numpy as np
import tensorflow as tf
from model.utils import read_yaml
from typing import List, Dict, Tuple


class RNNlayer:
    """
    Abstract class for RNN layers used by the HERMES model.
    """

    def __init__(self) -> None:
        """
        Instantiate a RNN layer
        """
        self._model_is_built = False

    def build(self, layer_name_to_shape: Dict) -> None:
        """
        Define the tf.keras.Input layers and name of the inputs
            
        Arguments:
        
        - *layer_name_to_shape*: Dict gathering information on the inputs of the RNN layer.
        """
        self.build_layer_names_to_shape(layer_name_to_shape)
        self.inputs_name = list(self.input_name_to_layer.keys())
        self._model_is_built = True

    def build_layer_names_to_shape(self, layer_name_to_shape: Dict) -> None:
        """
        Define the tf.keras.Input layers
        
        Arguments:
        
        - *layer_name_to_shape*: Dict gathering information on the inputs of the RNN layer.
        """

        self.input_name_to_layer = {}
        cont_input_shape = (layer_name_to_shape["window"], 1)
        for input_name in layer_name_to_shape["main_signal"]:
            self.input_name_to_layer[input_name] = tf.keras.layers.Input(
                cont_input_shape, name=input_name
            )
        if "temporal" in layer_name_to_shape["weak_signals"]:
            for input_name in layer_name_to_shape["weak_signals"]["temporal"]:
                self.input_name_to_layer[input_name] = tf.keras.layers.Input(
                    cont_input_shape, name=input_name
                )

        layer_name_to_shape["model_name"] = self.__class__.name


class ContLSTM(RNNlayer):
    """
    Class defining a RNN layer that takes several continuous inputs and process them with LSTM layers.
    """

    name = "cont_lstm"

    def __init__(self, option_file: Dict = None, **kwargs) -> None:
        """
        Build the architecture of the RNN layer based on a dict gathering the configartion of the model.
        
        Arguments:
        
        -*option_file*: Dict gathering information on the architecture of the RNN layer.
        """

        super().__init__(**kwargs)
        self._option_file = option_file
        if option_file is not None:
            self.process_option_file(option_file)

    def process_option_file(self, option_dict: Dict) -> None:
        """
        Process the option file and fill the argument self._lstm_layers
        
        Arguments:
        
        -*option_file*: Dict gathering information on the architecture of the RNN layer.
        """
        self._lstm_layers = [v for k, v in option_dict["lstm"].items()]

    def build(self, layer_name_to_shape: Dict) -> Tuple:
        """
        Build the RNN layer
        
        Arguments:
        
        - *layer_name_to_shape*: Dict gathering information on the inputs of the RNN layer.
        
        Returns:
        
        -*(self.model, self.inputs_name)* A Tuple with 2 elements: i) a tf.keras.Model ii) a list with the name of the different inputs.
        """
        super().build(layer_name_to_shape)
        if len(self.input_name_to_layer.keys()) == 1:
            lstm_outputs = list(self.input_name_to_layer.values())
        else:
            lstm_outputs = [
                tf.keras.layers.Concatenate()(
                    [inputs for inputs in self.input_name_to_layer.values()]
                )
            ]

        for i, lstm_layer_dict in enumerate(self._lstm_layers):
            lstm_layer = tf.keras.layers.LSTM(**lstm_layer_dict)
            lstm_output = lstm_layer(lstm_outputs[-1])
            lstm_outputs.append(lstm_output)

        model_output = tf.keras.layers.Dense(
            layer_name_to_shape["horizon"], name="prediction"
        )(lstm_outputs[-1])

        self.model = tf.keras.Model(
            inputs=self.input_name_to_layer, outputs=[model_output]
        )

        return self.model, self.inputs_name


name_to_archi = {
    ContLSTM.name: ContLSTM,
}

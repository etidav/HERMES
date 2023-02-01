import argparse
import os
import numpy as np
import tensorflow as tf
import pandas as pd

import sys
sys.path.append('/home/edavid/HERMES/')
sys.path.append('/home/edavid/HERMES/model/')

from model.hermes import hermes
from model.utils import read_pickle, write_pickle, read_yaml, write_yaml, read_json, write_json, tqdm



def main():
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.set_logical_device_configuration(
                    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=7500)]
                )
    os.environ['PYTHONHASHSEED']=str(42)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)
    
    y_signal = pd.read_csv('/home/edavid/HERMES/data/f1_main_100_sequences.csv', index_col=0)
    external_signal = pd.read_csv('/home/edavid/HERMES/data/f1_fashion_forward_100_sequences.csv', index_col=0)
    y_signal_train =  y_signal.iloc[:-52]
    external_signal_train =  external_signal.iloc[:-52]
    
    model_folder = os.path.join('/home/edavid/HERMES/result/hermes_model/',model_dir_tag)
    
    model_name = 'cont_lstm'
    input_model_config_path = '/home/edavid/HERMES/model/input_model_config/main_signal_ratioff_window_104.yaml'
    deep_model_config_path = '/home/edavid/HERMES/model/deep_model_config/3lstm50.yaml'
    stat_model_name = 'tbats'
    
    model = hermes(
        model_name=model_name,
        deep_model_config_path=deep_model_config_path,
        input_model_config_path=input_model_config_path,
        stat_model_name = stat_model_name,
        stat_model = None
    )
    
    batch_size = 1
    val_size = 52
    nb_window = 1
    early_stopping = 100
    rnn_lr = 0.005
    rnn_optimizer = tf.keras.optimizers.Adam(learning_rate=rnn_lr)

    model.fit(
        y_signal=y_signal_train,
        external_signal=external_signal_train,
        input_signature=input_signature,
        early_stopping=early_stopping,
        batch_size=batch_size,
        val_size=val_size,
        nb_window=nb_window,
        model_folder=model_folder,
        rnn_optimizer=rnn_optimizer,
        nb_max_epoch=nb_max_epoch,
    )
    
    hermes_prediction, stat_model_prediction = model.predict(y_signal_train, external_signal_train)

    hermes_eval = model.eval(self, ground_truth=y_signal, prediction=hermes_prediction, metrics = ['mase'])
    stat_model_eval = model.eval(self, ground_truth=y_signal, prediction=stat_model_prediction, metrics = ['mase'])
    
    print(f'hermes mase: {round(hermes_eval['mase'],3)}')
    print(f'{stat_model_name} mase: {round(stat_model_eval['mase'],3)}')
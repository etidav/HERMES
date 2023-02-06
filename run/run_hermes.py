import argparse
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from model.hermes import hermes
from tqdm import tqdm
from model.utils import read_pickle, write_pickle, read_yaml, write_yaml, read_json, write_json



def main():
    
    parser = argparse.ArgumentParser(description="train a HERMES model on a dataset of 100 time series")
    parser.add_argument(
        "--model_dir_tag", type=str, help="Name of the directory where the model will be store", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, help="size of the batch used during the training", default=8
    )
    parser.add_argument(
        "--val_size", type=int, help="size of the validation set", default=52
    )
    parser.add_argument(
        "--nb_window", type=int, help="number of windows that will be used as the training set", default=1
    )
    parser.add_argument(
        "--nb_max_epoch", type=int, help="maximum epochs before stopping the training", default=1000
    )
    parser.add_argument(
        "--early_stopping", type=int, help="number of epochs before stopping the training if no improvement of the loss function", default=100
    )
    parser.add_argument(
        "--rnn_lr", type=float, help="initial value of the optimizer's learning rate", default=0.0005
    )
    parser.add_argument(
        "--stat_model_name", type=str, help="name of the statistical model. Two models already implemented: ['ets', 'tbats']", default='tbats'
    )
    parser.add_argument(
        "--load_pretrain_stat_model", type=bool, help="Set to True if you want to load pretrained statistical models", default=True
    )
    args = parser.parse_args()
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus):
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.set_logical_device_configuration(
                        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=7500)]
                    )
    
    y_signal = pd.read_csv('/hermes/data/f1_main_100_sequences.csv', index_col=0)
    external_signal = pd.read_csv('/hermes/data/f1_fashion_forward_100_sequences.csv', index_col=0)
    y_signal_train =  y_signal.iloc[:-52]
    external_signal_train =  external_signal.iloc[:-52]
    
    model_name = 'cont_lstm'
    input_model_config_path = '/hermes/model/input_model_config/main_signal_ratioff_window_104.yaml'
    deep_model_config_path = '/hermes/model/deep_model_config/3lstm50.yaml'
    stat_model_name = args.stat_model_name
    
    batch_size = args.batch_size
    val_size = args.val_size
    nb_window = args.nb_window
    nb_max_epoch = args.nb_max_epoch
    early_stopping = args.early_stopping
    rnn_lr = args.rnn_lr
    rnn_optimizer = tf.keras.optimizers.Adam(learning_rate=rnn_lr)
    
    load_pretrain_stat_model = args.load_pretrain_stat_model
    pretrain_stat_model_train = read_pickle('/hermes/data/pretrain_stat_model/pretrain_stat_model_train.pkl')
    pretrain_stat_model_test = read_pickle('/hermes/data/pretrain_stat_model/pretrain_stat_model_test.pkl')
    
    all_mase = []
    
    for seed in range(10):
        
        model_dir_tag = args.model_dir_tag + f'_seed{seed}'
        model_folder = os.path.join('/hermes/result/',model_dir_tag)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)    

        os.environ['PYTHONHASHSEED']=str(seed)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        model = hermes(
            model_name=model_name,
            deep_model_config_path=deep_model_config_path,
            input_model_config_path=input_model_config_path,
            stat_model_name = stat_model_name,
            stat_model = None
        )

        if load_pretrain_stat_model:
            model.stat_model.stat_model = pretrain_stat_model_train

        model.fit(
            y_signal=y_signal_train,
            external_signal=external_signal_train,
            early_stopping=early_stopping,
            batch_size=batch_size,
            val_size=val_size,
            nb_window=nb_window,
            model_folder=model_folder,
            rnn_optimizer=rnn_optimizer,
            nb_max_epoch=nb_max_epoch,
        )

        if load_pretrain_stat_model:
            model.stat_model.stat_model = pretrain_stat_model_test

        hermes_prediction, stat_model_prediction = model.predict(y_signal_train, external_signal_train)


        hermes_eval = model.evaluate(ground_truth=y_signal, prediction=hermes_prediction, metrics = ['mase'])
        stat_model_eval = model.evaluate(ground_truth=y_signal, prediction=stat_model_prediction, metrics = ['mase'])
        hermes_mase = round(hermes_eval['mase'],3)
        stat_mase = round(stat_model_eval['mase'],3)
        print(f'hermes mase: {hermes_mase}')
        print(f'{stat_model_name} mase: {stat_mase}')
        
        all_mase.append(hermes_mase)
        
        write_json({'hermes':hermes_mase,'stat':stat_mase}, os.path.join(model_folder,'final_metrics.json'))
        
    
    print('HERMES final resul')
    print('mean ',np.mean(all_mase))
    print('std ',np.std(all_mase))
    
        
if __name__ == "__main__":

    main()
import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
import copy
from typing import List, Dict, Tuple
from tqdm import tqdm
from model.utils import (
    read_json,
    read_yaml,
    write_yaml,
)

from model.stat_layer import Statlayer
from model.rnn_layer import name_to_archi

random.seed(42)


class hermes(tf.keras.Model):
    """
    Class defining the a HERMES model.
    """

    def __init__(
        self,
        model_name: str = None,
        deep_model_config_path: str = None,
        input_model_config_path: str = None,
        model_folder: str = None,
        stat_model_name: str = None,
        stat_model: Dict = None,
    ) -> None:
        """
        Instantiate a HERMES model
        
        Arguments:

        - *model_name*: name of a RNNlayer.
        - *deep_model_config_path*: path to a yaml file containing a dict with the rnn architecture and hidden_layer size.
        - *input_model_config_path*: str : path to a yaml file containing a dict with the seasonality/window/horizon length and inputs/outputs name and shape.
        - *model_folder*: path to a directory containing a dcg model already train and save --> reload this model
        - *stat_model_name*: name of a statistical model that will be used in the Statlayer
        - *stat_model*: A dict gathering the statistical models already fiited. As the learning of the statistical models could be long, during a grid search, it could be useful to train only one time them and reload them at each new training of the HERMES model.
        """

        super(hermes, self).__init__()
        self.model_folder = model_folder
        if model_folder is not None:
            if any(
                [
                    model_name is not None,
                    deep_model_config_path is not None,
                    input_model_config_path is not None,
                    es_model is not None,
                    stat_model_name is not None,
                ]
            ):
                raise RuntimeError(
                    "if you specify a model_folder, don't specify any others arguments. All are going to be instanciate with the model_folder."
                )
            config = read_json(os.path.join(model_folder, "config.json"))
            input_model_config_path = os.path.join(model_folder, "input_signature.yaml")
            deep_model_config_path = os.path.join(
                model_folder, "deep_model_config.yaml"
            )
            self.process_input_model_config(input_model_config_path)
            self.model_name = config["model_name"]
            ref_model = hermes(
                model_name=self.model_name,
                deep_model_config_path=deep_model_config_path,
                input_model_config_path=input_model_config_path,
                stat_model_name=config["stat_model_name"],
            )
            ckptdir = os.path.join(model_folder, "ckpt")
            root = tf.train.Checkpoint(model=ref_model.deep_model)
            root.restore(tf.train.latest_checkpoint(ckptdir))
            self.deep_model = ref_model.deep_model
            self.deep_model.signature = ref_model.deep_model.signature
            self.stat_model = ref_model.stat_model
        else:
            if any(
                [
                    model_name is None,
                    deep_model_config_path is None,
                    input_model_config_path is None,
                    stat_model_name is None,
                ]
            ):
                raise RuntimeError(
                    "if you don't specify a model_folder, you must specify all others arguments :  model_name, deep_model_config_path, input_model_config_path, stat_model_name"
                )
            self.input_model_config = read_yaml(input_model_config_path)
            self.deep_model_config = read_yaml(deep_model_config_path)
            self.process_input_model_config()
            self.model_name = model_name
            self.deep_model, self.deep_model.signature = name_to_archi[self.model_name](
                self.deep_model_config
            ).build(self.input_model_config)
            self.stat_model = Statlayer(
                seasonality=self.seasonality,
                window=self.window,
                horizon=self.horizon,
                stat_model_name=stat_model_name,
                stat_model=stat_model,
            )

    def process_input_model_config(self) -> None:
        """
        Instantiate self.seasonality, self.horizon and self.window
        """
        self.seasonality = self.input_model_config["seasonality"]
        self.horizon = self.input_model_config["horizon"]
        self.window = self.input_model_config["window"]

    def call(self, inputs: Dict, training: bool = False) -> tf.Tensor:
        """
        compute the (non-renormlazed) output of the HERMES model.
        
        Arguments:
        
        -*inputs*: a dict containing all the inputs of the HERMES model
        
        - *training*: A boolean indicating if the function is used during the training process or during the inference process.
        
        Returns:
        
        -*pred_windows*: a tf.Tensor with the output of the HERMES model
        """
        values_X = inputs["trends"]

        if training:
            input_windows = {}

            input_scaled = tf.reshape(
                tf.math.reduce_mean(
                    self.make_moving_windows(
                        trends=inputs["trends"],
                        window=self.window,
                        horizon=self.horizon,
                    ),
                    axis=1,
                ),
                (-1, 1),
            )
            for cont_input in self.deep_model.signature:
                if cont_input == "trends":
                    input_window = self.make_moving_windows(
                        trends=values_X,
                        window=self.window,
                        horizon=self.horizon,
                    )

                    stat_preds = self.stat_model(inputs)

                    if self.window == 104:
                        stat_preds_input = tf.tile(stat_preds, [1, 2])
                        input_window = tf.math.add(input_window, -stat_preds_input)

                    else:
                        input_window = tf.math.add(input_window, -stat_preds)

                    input_window = tf.math.divide(input_window, input_scaled)

                    input_windows[cont_input] = tf.expand_dims(input_window, axis=-1)

                else:
                    input_window = self.make_moving_windows(
                        trends=inputs[cont_input],
                        window=self.window,
                        horizon=self.horizon,
                    )

                    input_windows[cont_input] = tf.expand_dims(input_window, axis=-1)

            stat_preds = self.stat_model(inputs)
            pred_windows = self.deep_model(input_windows)

            pred_windows_rescaled = tf.math.multiply(pred_windows, input_scaled)

            model_pred = tf.math.add(pred_windows_rescaled, stat_preds)
            y_true = self.make_moving_windows(
                trends=inputs["trends"][:, self.window :],
                window=self.horizon,
                horizon=0,
            )
            err_deep = tf.math.reduce_sum(
                tf.math.abs(tf.math.add(y_true, -model_pred)), axis=1
            )
            deep_loss = tf.math.divide(err_deep, tf.squeeze(input_scaled))

        else:  # using only the last window to do a prediction

            input_scaled = tf.reshape(
                tf.math.reduce_mean(inputs["trends"][:, -self.window :], axis=1),
                (-1, 1),
            )
            input_windows = {}
            for cont_input in self.deep_model.signature:
                if cont_input == "trends":

                    input_window = values_X[:, -self.window :]

                    stat_preds = self.stat_model(inputs, inference=True)

                    if self.window == 104:
                        stat_preds_input = tf.tile(stat_preds, [1, 2])
                        input_window = tf.math.add(input_window, -stat_preds_input)
                    else:
                        input_window = tf.math.add(input_window, -stat_preds)
                    input_window = tf.math.divide_no_nan(input_window, input_scaled)

                    input_windows[cont_input] = input_window

                elif cont_input == "stat_pred":
                    stat_preds_input = self.stat_model(inputs, inference=True)
                    if self.window == 104:
                        stat_preds_input = tf.tile(stat_preds_input, [1, 2])
                    input_window = tf.math.divide_no_nan(stat_preds_input, input_scaled)
                    input_window = tf.expand_dims(input_window, axis=-1)
                    input_windows[cont_input] = input_window
                else:
                    input_window = tf.expand_dims(
                        inputs[cont_input][:, -self.window :], -1
                    )
                    input_windows[cont_input] = input_window

            stat_preds = self.stat_model(inputs, inference=True)
            pred_windows = self.deep_model(input_windows)
            pred_windows = tf.math.multiply(pred_windows, input_scaled)
            pred_windows = tf.math.add(pred_windows, stat_preds)
            deep_loss = 0.0

        self.add_loss(deep_loss)
        return pred_windows

    def predict(
        self,
        y_signal: pd.DataFrame,
        external_signal: pd.DataFrame,
        stat_model: str = None,
        date_time: str = None,
    ) -> Tuple:
        """
        compute the final prediction of the HERMES model.
        
        Arguments:
        
        - *y_signal*: a pd.DataFrame with one or multiple time series.
        - *external_signal*: a pd.DataFrame with the linked external signals
        - *stat_model*: A dict gathering the statistical models already fiited. As the learning of the statistical models could be long, during a grid search, it could be useful to train only one time them and reload them at each new training of the HERMES model.
        - *date_time*: a date time indicating where to start the HERMES prediction (format YYYY-MM-DD)
        
        Returns:
        
        -*(final_prediction, stat_prediction)*: a Tuple with two pd.DataFrame: i) the final prediction of the hermes model ii) the intermediate prediction of the per time series statistical models 
        """

        if self.model_folder is None:
            raise RuntimeError(
                "dcg.model_folder need to be define, load dcg from a model_folder or train model before predict"
            )

        y_signal_train = copy.deepcopy(y_signal)
        external_signal_train = copy.deepcopy(external_signal)

        if date_time is not None:
            y_signal_train = y_signal_train.loc[:date_time]
            external_signal_train = external_signal_train.loc[:date_time]

        if stat_model is not None:
            self.stat_model.stat_model = stat_model
            
        processes=os.cpu_count() - 2

        if not self.stat_model.check_integrity(y_signal_train, train=False):
            self.stat_model.fit(y_signal_train, train=False, processes=processes)
        self.stat_model.compute_prediction(y_signal_train, train=False)

        inputs, trends_idx = self.sequences_to_model_inputs(
            y_signal_train,
            external_signal_train,
            tf_dataset=False,
        )

        data = y_signal_train

        model_output = self(inputs)

        preds = model_output.numpy().T

        ts_name = y_signal_train.columns

        delta = pd.to_datetime(y_signal_train.index[-1]) - pd.to_datetime(
            y_signal_train.index[-2]
        )
        index = [
            str((pd.to_datetime(y_signal_train.index[-1]) + delta * (i + 1)).date())
            for i in range(self.horizon)
        ]

        final_prediction = pd.DataFrame(preds, columns=ts_name, index=index)

        stat_prediction = pd.DataFrame(
            self.stat_model(inputs, inference=True).numpy().T,
            columns=ts_name,
            index=index,
        )

        return final_prediction, stat_prediction

    def sequences_to_model_inputs(
        self,
        y_signal: pd.DataFrame,
        external_signal: pd.DataFrame,
        tf_dataset: bool = True,
    ) -> Tuple:
        """
        Transform pd.DataFrame containing the main and external signals into a Dict that will be used as input by the HERMES model.
        
        Arguments:

        - *y_signal*: a pd.DataFrame with one or multiple time series.
        - *external_signal*: a pd.DataFrame with the linked external signals
        - *tf_dataset*: bool, if set True, output is shuffled at each call

        Returns:

        - *(all_inputs, trend_idx)*: if tf_dataset == True, return a Tuple with 2 elements: i) a dict of tf.data.Dataset, ii) a np.array with the index of the time series. if tf_dataset == False, the first element is a dict of np.array
        """
        all_inputs = {}

        trends_idx = pd.DataFrame(
            range(len(y_signal.columns)), index=y_signal.columns, columns=["batch_idx"],
        )
        list_trends_name = list(y_signal.columns).copy()
        
        if tf_dataset:

            random.shuffle(list_trends_name)

            all_inputs["trends"] = tf.data.Dataset.from_tensor_slices(
                y_signal[list_trends_name].values.T.astype("float32")
            )

            all_inputs["trends_idx"] = tf.data.Dataset.from_tensor_slices(
                trends_idx.loc[list_trends_name].values.astype("int32")
            )

            if "temporal" in self.input_model_config["weak_signals"].keys():
                for key in self.input_model_config["weak_signals"]["temporal"]:
                    if key == "ratio_fashion_forwards":
                        ratio_fashion_forwards = external_signal / (
                            y_signal + external_signal
                        )
                        ratio_fashion_forwards = ratio_fashion_forwards.fillna(0.5)
                        all_inputs[key] = tf.data.Dataset.from_tensor_slices(
                            ratio_fashion_forwards[list_trends_name].values.T.astype(
                                "float32"
                            )
                        )
                    else:
                        print(f"{key} not recognized")

        else:

            all_inputs["trends"] = y_signal.values.T.astype("float32")

            all_inputs["trends_idx"] = trends_idx.values.astype("int32")

            if "temporal" in self.input_model_config["weak_signals"].keys():
                for key in self.input_model_config["weak_signals"]["temporal"]:
                    if key == "ratio_fashion_forwards":
                        ratio_fashion_forwards = external_signal / (
                            y_signal + external_signal
                        )
                        ratio_fashion_forwards = ratio_fashion_forwards.fillna(0.5)
                        all_inputs[key] = ratio_fashion_forwards.values.T.astype(
                            "float32"
                        )
                    else:
                        print(f"{key} not recognized")

        return all_inputs, trends_idx.loc[list_trends_name].values.astype("int32")

    def make_moving_windows(
        self, trends: tf.Tensor, window: int, horizon: int
    ) -> tf.Tensor:
        """
        apply a moving window to the time series to increase the size of the training set
        
        Arguments:

        - *trends*: a tf.Tensor containing the values of the different time series.
        - *window*: Define the look back window the of HERMES model.
        - *horizon*: Define the horizon of the forecast of the statistical model

        Returns:

        - *moving_windows*: a tf.Tensor containing the sequences generated from the inputs using a moving window of size window
        """
        moving_windows = [
            trends[:, i : i + window]
            for i in range(trends.shape[1] - window - horizon + 1)
        ]

        moving_windows = tf.concat(moving_windows, axis=0)
        return moving_windows

    @tf.function
    def grad(self, inputs: Dict, val_size: int, nb_window: int) -> Dict:
        """
        Compute the loss of the model and the correspond gradients
        
        Arguments:

        - *inputs*: dict containing all the HERMES inputs.
        - *val_size*: size of the validation set.
        - *nb_window*: How many moving windows are used in the training process.

        Returns:

        - *losses*: dict with the loss function in train and val of the model and the linked gradients
        """

        batch_size = inputs["trends"].shape[0]
        timesteps = inputs["trends"].shape[1]
        train_size = timesteps - self.horizon - self.window + 1 - val_size
        if nb_window == 0:
            nb_window = train_size

        if train_size < 1:
            raise ValueError(
                "Empty training set. Timesteps {}, horizon {}, window {} and val_size {}".format(
                    timesteps, self.horizon, self.window, val_size
                )
            )

        with tf.GradientTape() as tape:
            model_output = self(inputs, training=True)
            if np.array([len(loss.shape) == 0 for loss in self.losses]).any():
                raise ValueError(
                    "Invalid shape for loss. Make sure you called model in training mode."
                )
            losses = {}

            assert self.losses[0].shape[0] == (train_size + val_size) * batch_size

            train_loss = self.losses[0][
                batch_size * (train_size - nb_window) : batch_size * train_size
            ]
            losses["train_loss"] = tf.math.reduce_mean(train_loss)
            val_loss = self.losses[0][-batch_size:]
            losses["val_loss"] = tf.math.reduce_mean(val_loss)

        grads = {
            "grad": tape.gradient(
                losses["train_loss"], self.deep_model.trainable_variables
            )
        }
        losses.update(grads)

        return losses

    def fit(
        self,
        y_signal: pd.DataFrame,
        external_signal: pd.DataFrame,
        batch_size: int,
        val_size: int,
        nb_window: int,
        model_folder: str,
        early_stopping: int = None,
        rnn_reducelr: int = None,
        rnn_optimizer: tf.keras.optimizers.Optimizer = None,
        nb_max_epoch: int = 20,
        return_last_ckpt: bool = True,
    ) -> None:

        """
        Fit a HERMES model
        
        Arguments:

        - *y_signal*: a pd.DataFrame with one or multiple time series.
        - *external_signal*: a pd.DataFrame with the linked external signals
        - *batch_size*: size of the batch during training
        - *val_size*: number of window to hide during the training
        - *nb_window*: How many moving windows are used in the training process.
        - *model_folder*: str, where to store the final model and checkpoint
        - *early_stopping*: int, number of epoch without betterment before stop training
        - *rnn_reducelr*: number of epoch without betterment before decrease the rnn_optimizer learning
        - *rnn_optimizer*: tf.keras.optimizer use to updtate the RNN variables
        - *nb_max_epoch*:  max number of epoch during the training
        - *return_last_ckpt*: boolean to indicate if the weights of the best epoch are returned or if the weights of the last epoch are returned        
       """

        if rnn_optimizer is None:
            raise ValueError("rnn_optimizer should be define")
        if not os.path.exists(model_folder):
            raise ValueError("directory not found, provide a correct model_folder")
        else:
            self.model_folder = model_folder
            write_yaml(
                self.input_model_config,
                os.path.join(self.model_folder, "input_signature.yaml"),
            )
            write_yaml(
                self.deep_model_config,
                os.path.join(self.model_folder, "deep_model_config.yaml"),
            )

        logdir = os.path.join(model_folder, "log/")
        writer = tf.summary.create_file_writer(logdir)

        ckptdir = os.path.join(model_folder, "ckpt/")

        root = tf.train.Checkpoint(model=self.deep_model)
        tf_manager = tf.train.CheckpointManager(root, directory=ckptdir, max_to_keep=1)

        if not self.stat_model.check_integrity(
            y_signal, val_size=val_size, nb_window=nb_window, train=True
        ):
            self.stat_model.fit(
                y_signal,
                val_size=val_size,
                nb_window=nb_window,
                train=True,
                processes=processes,
            )
        self.stat_model.compute_prediction(
            y_signal, val_size=val_size, nb_window=nb_window, train=True
        )

        deep_loss = np.inf
        val_loss = np.inf
        rnn_optimizer_lr = rnn_optimizer.lr.numpy()
        early_stopping_count = 0

        all_inputs, trends_idx = self.sequences_to_model_inputs(
            y_signal=y_signal, external_signal=external_signal
        )

        with writer.as_default():
            for epoch in tqdm(range(nb_max_epoch)):

                early_stopping_count += 1

                train_loss_list = []
                val_loss_list = []

                all_inputs_batch = tuple(
                    zip(
                        *[
                            all_inputs[inputs_name].batch(batch_size)
                            for inputs_name in all_inputs.keys()
                        ]
                    )
                )

                if epoch != 0:
                    for i, batch_input in enumerate(all_inputs_batch):

                        inputs = {}
                        for k, input_name in enumerate(all_inputs.keys()):
                            inputs[input_name] = batch_input[k]

                        dict_grad = self.grad(inputs, val_size, nb_window)

                        rnn_optimizer.apply_gradients(
                            zip(dict_grad["grad"], self.deep_model.trainable_variables)
                        )

                for i, batch_input in enumerate(all_inputs_batch):

                    inputs = {}
                    for k, input_name in enumerate(all_inputs.keys()):
                        inputs[input_name] = batch_input[k]

                    dict_grad = self.grad(inputs, val_size, nb_window)

                    train_loss_list.append(dict_grad["train_loss"])
                    val_loss_list.append(dict_grad["val_loss"])

                epoch_train_loss = np.array(train_loss_list).mean()
                epoch_val_loss = np.array(val_loss_list).mean()

                tf.summary.scalar("Train_em_loss", epoch_train_loss, step=epoch)
                tf.summary.scalar("Val_em_loss", epoch_val_loss, step=epoch)

                print(
                    "rnn epoch {:03d}: Train_loss {:.8f}  Val_loss {:.8f}".format(
                        epoch, epoch_train_loss, epoch_val_loss
                    )
                )

                if epoch_val_loss < deep_loss:
                    deep_loss = epoch_val_loss
                    early_stopping_count = 0
                    print(
                        "Loss change at epoch {:03d}: {:.8f}".format(
                            epoch, epoch_val_loss
                        )
                    )
                    tf_manager.save()

                    if early_stopping_count >= 40 and not early_stopping_count % 10:
                        rnn_optimizer.lr = rnn_optimizer.lr * 0.5
                        print(
                            "rnn_lr change at epoch {:03d}: {:.5f}".format(
                                epoch, rnn_optimizer.lr.numpy()
                            )
                        )

                if early_stopping is not None:
                    if early_stopping_count == early_stopping:

                        print(
                            "Early stopping : {:03d} epochs without betterment, Stop train at epoch {:03d}".format(
                                early_stopping_count, epoch
                            )
                        )
                        break

        if return_last_ckpt:
            root.restore(tf.train.latest_checkpoint(ckptdir))

    def evaluate(
        self,
        ground_truth: pd.DataFrame,
        prediction: pd.DataFrame,
        metrics: List = ["mase"],
    ) -> Dict:
        """
        Evaluate the prediction of a HERMES model
        
        Arguments:

        - *ground_truth*: a pd.DataFrame with the historical data and the ground truth of the forecasted timeframe 
        - *prediction*: a pd.DataFrame with the prediction of the model
        - *metrics*: A list containing a list of metrics to compute
        
        Returns:
        
        - *model_eval*: a dict containing the metrics values
       """
        time_split = prediction.index[0]
        histo_ground_truth = ground_truth.loc[:time_split].iloc[:-1]
        ground_truth = ground_truth.loc[time_split:].iloc[: self.horizon]
        

        if type(metrics) == str:
            metrics = [metrics]

        model_eval = {}
        for metric in metrics:
            if metric == "mase":
                denominator = np.mean(
                    np.abs(
                        histo_ground_truth.values[self.seasonality :]
                        - histo_ground_truth.values[: -self.seasonality]
                    ),
                    axis=0,
                )
                numerator = np.mean(
                    np.abs(ground_truth.values - prediction.values), axis=0
                )
                mase = (numerator / denominator).mean()
                model_eval["mase"] = mase
            else:
                print(
                    f"{metric} not implemented yet --> implemented metrics are {metrics}"
                )

        return model_eval

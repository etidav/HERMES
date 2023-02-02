import numpy as np
import pandas as pd
import tensorflow as tf
from tbats import TBATS
from statsmodels.tsa.api import ExponentialSmoothing
import multiprocessing
from functools import partial
from tqdm import tqdm
from typing import List, Dict


class Statlayer(tf.keras.layers.Layer):
    """
    Class defining a Stat layer used by the HERMES model.
    """

    def __init__(
        self,
        seasonality: int,
        horizon: int,
        window: int,
        stat_model_name: str,
        stat_model: Dict = None,
    ) -> None:
        """
        Instantiate a Stat layer used by the HERMES model.
        
        Arguments:
        
        - *seasonality*: Define the seasonality parameter of the statistical model that will be used in the HERMES apporach
        - *horizon*: Define the horizon of the forecast of the statistical model
        - *window*: Define the look back window the of HERMES model. usually equal to 1 of 2 time the seasonality
        - *stat_model_name*: name of the statistical model that will be used in the HERMES methode. Two statistical models are implemented for now: ['tbats', 'ets']
        - *stat_model*: A dict gathering the statistical models already fiited. As the learning of the statistical models could be long, during a grid search, it could be useful to train only one time them and reload them at each new training of the HERMES model.
        """
        super(Statlayer, self).__init__()
        self.seasonality = seasonality
        self.horizon = horizon
        self.window = window
        self.stat_model_name = stat_model_name
        if stat_model:
            self.stat_model = stat_model
        else:
            self.stat_model = {}

    def train_window_list(
        self, index_list: List, val_size: int, nb_window: int
    ) -> List:
        """
        Return a list of index indicating where to stop a time series and start to learn a statistical model depending of the moving window used in the HERMES model.
        
        Arguments:
        
        - *index_list*: A list gathering all the time index of a time series.
        - *val_size*: Size of the validation set.
        - *nb_window*: How many windows are needed by the HERMES model.
        
        Returns:
        
        - *train_list*: A list gathering time index that will be used during the training of the HERMES model
        """
        train_list = list(
            index_list[self.window - 1 : -self.horizon - val_size][-nb_window:]
        )
        train_list.append(index_list[-self.horizon - 1])
        return train_list

    def fit_model(
        self, y: pd.Series, val_size: int, nb_window: int, train: bool
    ) -> Dict:
        """
        Fit single statistical models on the time series y.
        
        Arguments:
        
        - *y*: a pd.Series with a single time series.
        - *val_size*: Size of the validation set.
        - *nb_window*: How many windows are needed by the HERMES model.
        - *train*: A boolean indicating if the function is used during the training process or during the inference process.
        
        Returns:
        
        - *stat_model*: A Dict gathering all the statistical models trained on the time series y.
        """
        y_name = y[0]
        stat_model = {y_name: {}}
        if self.stat_model_name == "tbats":
            model = TBATS(seasonal_periods=[self.seasonality], n_jobs=1)
        elif self.stat_model_name == "ets":
            model = ExponentialSmoothing(
                y[1].values, seasonal_periods=self.seasonality, seasonal="add"
            )

        if train:
            for date_time in self.train_window_list(y[1].index, val_size, nb_window):
                model_already_fitted = 0
                if y_name in self.stat_model:
                    if date_time in self.stat_model[y_name]:
                        if type(self.stat_model[y_name][date_time]) == type(model):
                            model_already_fitted += 1

                if model_already_fitted:
                    continue
                else:
                    if self.stat_model_name == "tbats":
                        fitted_model = model.fit(y[1].loc[:date_time])
                    elif self.stat_model_name == "ets":
                        model = ExponentialSmoothing(
                            y[1].loc[:date_time].values,
                            seasonal_periods=self.seasonality,
                            seasonal="add",
                        )
                        fitted_model = model.fit()
                    stat_model[y_name][date_time] = fitted_model
        else:
            date_time = y[1].index[-1]
            model_already_fitted = 0
            if y_name in self.stat_model:
                if date_time in self.stat_model[y_name]:
                    if type(self.stat_model[y_name][date_time]) == type(model):
                        model_already_fitted += 1

            if not model_already_fitted:
                if self.stat_model_name == "tbats":
                    fitted_model = model.fit(y[1].loc[:date_time])
                elif self.stat_model_name == "ets":
                    model = ExponentialSmoothing(
                        y[1].loc[:date_time].values,
                        seasonal_periods=self.seasonality,
                        seasonal="add",
                    )
                    fitted_model = model.fit()
                stat_model[y_name][date_time] = fitted_model
        return stat_model

    def fit(
        self,
        trends: pd.DataFrame,
        val_size: int = 0,
        nb_window: int = 0,
        train: bool = True,
        processes: int = 1,
    ) -> None:
        """
        Fit statistical models on all the time series of a dataset.
        
        Arguments:
        
        - *trends*: A DataFrame with one or multiple time series.
        - *val_size*: Size of the validation set.
        - *nb_window*: How many windows are needed by the HERMES model.
        - *train*: A boolean indicating if the function is used during the training process or during the inference process.
        - *processes*: Define how many CPU will be used to fi all the statistical models.
        """
        with multiprocessing.Pool(processes=processes) as pool:
            fitted_model = list(
                tqdm(
                    pool.imap(
                        partial(
                            self.fit_model,
                            val_size=val_size,
                            nb_window=nb_window,
                            train=train,
                        ),
                        trends.iteritems(),
                        chunksize=5,
                    )
                )
            )

        for i in fitted_model:
            self.stat_model.update(i)

    def check_integrity(
        self,
        trends: pd.DataFrame,
        val_size: int = 0,
        nb_window: int = 0,
        train: bool = True,
    ) -> bool:
        """
        Verify that all the statistical models are fitted.
        
        Arguments:
        
        - *index_list*: A list gathering all the time index of a time series.
        - *val_size*: Size of the validation set.
        - *nb_window*: How many windows are needed by the HERMES model.
        - *train*: A boolean indicating if the function is used during the training process or during the inference process.
        - *processes*: Define how many CPU will be used to fi all the statistical models.
        
        Returns:
        
        - *all_stat_model_fitted*: A boolean indicating if every statistical models are fitted.
        """
        all_stat_model_fitted = True
        if train:
            list_stat_model = self.stat_model.keys()
            for i in trends.columns:
                if i not in list_stat_model:
                    all_stat_model_fitted = False
                else:
                    for date_time in self.train_window_list(
                        trends.index, val_size, nb_window
                    ):
                        if date_time not in self.stat_model[i].keys():
                            all_stat_model_fitted = False
        else:
            list_stat_model = self.stat_model.keys()
            for i in trends.columns:
                if i not in list_stat_model:
                    all_stat_model_fitted = False
                else:
                    date_time = trends.index[-1]
                    if date_time not in self.stat_model[i].keys():
                        all_stat_model_fitted = False

        return all_stat_model_fitted

    def compute_prediction(
        self,
        trends: pd.DataFrame,
        val_size: int = 0,
        nb_window: int = 0,
        train: bool = True,
    ) -> None:
        """
        Compute the prediction of the statistical models on all the time series of a dataset and stock them in self.all_stat_pred.
        
        Arguments:
        
        - *trends*: A DataFrame with one or multiple time series.
        - *val_size*: Size of the validation set.
        - *nb_window*: How many windows are needed by the HERMES model.
        - *train*: A boolean indicating if the function is used during the training process or during the inference process.
        - *processes*: Define how many CPU will be used to fi all the statistical models.
        """

        check_integrity = self.check_integrity(
            trends, val_size=val_size, nb_window=nb_window, train=train
        )

        if check_integrity:

            all_stat_pred = {}

            for i in tqdm(trends):
                y = trends[i]

                if train:
                    train_window_list = self.train_window_list(
                        y.index, val_size, nb_window
                    )
                    for date_time in y.index[self.window - 1 : -self.horizon]:
                        if date_time in train_window_list:
                            model = self.stat_model[i][date_time]
                            forecast = model.forecast(self.horizon)
                        else:
                            forecast = np.repeat(0.0, self.horizon)

                        if date_time not in all_stat_pred:
                            all_stat_pred[date_time] = []
                        all_stat_pred[date_time].append(forecast)
                else:
                    date_time = y.index[-1]
                    model = self.stat_model[i][date_time]
                    forecast = model.forecast(self.horizon)
                    if date_time not in all_stat_pred:
                        all_stat_pred[date_time] = []
                    all_stat_pred[date_time].append(forecast)

            if train:
                for date_time in y.index[self.window - 1 : -self.horizon]:
                    all_stat_pred[date_time] = tf.constant(all_stat_pred[date_time])
                self.index_order = y.index[self.window - 1 : -self.horizon]
            else:
                date_time = y.index[-1]
                all_stat_pred[date_time] = tf.constant(all_stat_pred[date_time])
                self.index_order = [y.index[-1]]

            self.all_stat_pred = all_stat_pred

        else:
            raise RuntimeError(
                "Can't find all the stat models. Fit the stat models before use the call function"
            )

    def call(self, inputs: Dict, inference: bool = False) -> tf.Tensor:
        """
        Return the statistical predictions corresponding to the time series in the tf.Tensor inputs.
        
        Arguments:
        
        - *inputs*: a Dict containing all the inputs of the HERMES model.
        - *inference*: A boolean indicating if the function is used during the training process or during the inference process.
        
        Returns:
        
        - *batch_stat_pred*: a Tf.Tensor with all the statistical predictions.
        """

        if not inference:
            batch_stat_pred = tf.concat(
                [
                    tf.gather_nd(self.all_stat_pred[date_time], inputs["trends_idx"])
                    for date_time in self.index_order
                ],
                axis=0,
            )
        else:
            batch_stat_pred = tf.gather_nd(
                self.all_stat_pred[self.index_order[-1]], inputs["trends_idx"]
            )
        batch_stat_pred = tf.cast(batch_stat_pred, tf.float32)
        return batch_stat_pred

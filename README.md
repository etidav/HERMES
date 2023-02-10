# HERMES: Hybrid Error-corrector Model with inclusion of External Signals for nonstationary fashion time series

Authors: Etienne DAVID, Jean BELLOT and Sylvain LE CORFF

Paper link: 

### Abstract
> Developing models and algorithms to draw causal inference for time series is a long standing statistical problem. It is crucial for many applications, in particular for fashion or retail industries, to make optimal inventory decisions and avoid massive wastes. By tracking thousands of fashion trends on social media with state-of-the-art computer vision approaches, we propose a new model for fashion time series forecasting. Our contribution is  twofold. We first provide [publicly](http://files.heuritech.com/raw_files/f1_fashion_dataset.tar.xz) an appealing fashion dataset gathering 10000 weekly fashion time series. As influence dynamics are the key of emerging trend detection, we associate with each time series an external weak signal representing behaviours of influencers. Secondly, to leverage such a complex and rich dataset, we propose a new hybrid forecasting model. Our approach combines per-time-series parametric models with seasonal components and a global recurrent neural network to include sporadic external signals. This hybrid model provides state-of-the-art results on the proposed fashion dataset, on the weekly time series of the M4 competition, and illustrates the benefit of the contribution of external weak signals.

## Code Organisation

This repository provides the code of the HERMES approach and a simple code base to reproduce the results presented in the [paper](https://arxiv.org/pdf/2202.03224.pdf). The repository is organized as follow:

 - [model/](model/): Directory gathering the code of HERMES
 - [run/](run/): Directory containing a script to reproduce the HERMES result of the paper.
 - [data/](data/): Directory gathering 100 fashion time series introduced in the paper [paper](https://arxiv.org/pdf/2202.03224.pdf) and used in the experimental section, Table 4.
 - [docker/](docker/): directory gathering the code to reproduce a docker so as to recover the exact result of the paper. 

## Reproduce benchmark results

First, you should build, run and enter into the docker. In the main folder, run
```bash
make build run enter
```

To reproduce the result of HERMES on the sample of 100 time series:
- [run_hermes.py](run/run_hermes.py)
run
```bash
python run/run_hermes.py --help # display the default parameters and their description
python run/run_hermes.py --model_dir_tag hermes_100ts --nb_time_series 100 --rnn_lr 0.001 --batch_size 8 --load_pretrain_stat_model # train 10 hermes with different seeds on the sample of 100 fashion time series and save the results in the dir result/
python run/run_hermes.py --model_dir_tag hermes_1000ts --nb_time_series 1000 --rnn_lr 0.0005 --batch_size 64 --load_pretrain_stat_model # train 10 hermes with different seeds on the sample of 1000 fashion time series and save the results in the dir result/
python run/run_hermes.py --model_dir_tag hermes_10000ts --nb_time_series 10000 --rnn_lr 0.001 --batch_size 64 --load_pretrain_stat_model # train 10 hermes with different seeds on the whole fashion time series and save the results in the dir result/
```

## HERMES paper results on 100ts

The following tabs summarize some results that can be reproduced with this code:


 - Experience on 10000 time series:

| Model         | Mase  mean  | Mase std    |
| :-------------| :-----------| :-----------|
| tbats         |    0.745    |    -        |
| hermes-tbats  |             |             |

 - Experience on 1000 time series:

| Model         | Mase  mean  | Mase std    |
| :-------------| :-----------| :-----------|
| tbats         |    0.734    |    -        |
| hermes-tbats  |    0.7151   |    0.0035   |

 - Experience on 100 time series:

| Model         | Mase  mean  | Mase std    |
| :-------------| :-----------| :-----------|
| tbats         |    0.745    |    -        |
| hermes-tbats  |    0.7368   |    0.0043   |

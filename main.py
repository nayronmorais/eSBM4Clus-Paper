import sys

import numpy as np
import pandas as pd

from evos.datasets.datasource import CollectionTabularData, TabularData

from evos.clustering import eSBM4Clus
from evos.clustering import CEDAS
from evos.clustering import MacroSOStream
from evos.clustering import SOStream
from evos.clustering import AutoCloud
from evos.clustering import MicroTEDAClus
from evos.clustering import OEC
from evos.clustering import MCMSTStream

from evos.clustering.river import DenStream
from evos.clustering.river import DBSTREAM

from evos.benchmarking import SearchThenPreq

from river.metrics import AdjustedRand

from helpers import linspace, bfill, ffill, normalization, add_noise


random_state = 1

gpreq = SearchThenPreq( 
        clear_results=False,
        save_path='results',
        verbose=2,
        search_kwargs={
            'cv': 0,
            'scoring': 'adjusted_rand_score',
            'verbose': 0,
            'n_jobs': 30,
            'pre_dispatch': 20,
            'n_iter': 100, 
            'random_state': random_state
        }
 )


### ------------------------------- ADDING METHODS --------------------------------------------###

n_params = 10
precision = 3



## ------------------------
# eSBM4Clus

esbm4clus_params_grid = dict(
    w=linspace(5, 50, num=n_params, dtype=np.int64),   
    beta=linspace(0.001, 0.999, num=n_params, dtype=float, precision=precision),
)

gpreq.add_method("eSBM4Clus", eSBM4Clus(l_max=300, gamma=0.1), param_grid=esbm4clus_params_grid)

## -------------------------


## ------------------------
## CEDAS

cedas_params_grid = dict(
    min_samples=linspace(2, 20, num=n_params, dtype=np.int64),
    radius=linspace(0.001, 0.5, num=n_params, dtype=float, precision=precision),
    decay=linspace(100, 2000, num=n_params, dtype=np.int64)
)

gpreq.add_method("CEDAS", CEDAS(), param_grid=cedas_params_grid)

# -------------------------


## ------------------------
## AutoCloud

auto_params_grid = dict(
    m=linspace(1.0, 10.0, num=n_params, dtype=float, precision=precision)
)

gpreq.add_method("AutoCloud", AutoCloud(), param_grid=auto_params_grid)

# -------------------------


## ------------------------
## MicroTEDAClus

micro_params_grid = dict(
    r0=linspace(0.001, 0.5, num=n_params, dtype=float, precision=precision),
)

gpreq.add_method("MicroTEDAClus", MicroTEDAClus(), param_grid=micro_params_grid)

# -------------------------


# ------------------------
## OEC

oec_params_grid = dict(
    stab_period=linspace(0, 20, num=n_params, dtype=np.int64),
    gamma_normal=linspace(0.5, 0.999, num=n_params, dtype=float, precision=precision),
    ff=linspace(0.5, 0.99, num=n_params, dtype=float),
)

gpreq.add_method("OEC", OEC(), param_grid=oec_params_grid)

# -------------------------


# ------------------------
# MacroSOStream 

macro_params_grid = dict(
    min_pts=linspace(1, 20, num=n_params, dtype=np.int64),
    alpha=linspace(0.01, 0.9, num=n_params, dtype=float, precision=precision),
    merge_threshold=linspace(0.001, 0.5, num=n_params, dtype=float, precision=precision),
    p=linspace(1.0, 10.0, num=n_params, dtype=float, precision=precision)
)

gpreq.add_method("MacroSOStream", MacroSOStream(), param_grid=macro_params_grid)

# -------------------------


## ------------------------
## SOSTream

sos_params_grid = dict(
    alpha=linspace(0.01, 0.9, num=n_params, dtype=float, precision=precision),
    min_pts=linspace(2, 20, num=n_params, dtype=np.int64),
    merge_threshold=linspace(0.001, 0.5, num=n_params, dtype=float, precision=precision),
)

gpreq.add_method("SOStream", SOStream(merge=True, decay=False), param_grid=sos_params_grid)

# -------------------------


## ------------------------
## MCMSTStream

mcmc_params_grid = dict(
    sliding_window_size=linspace(100, 1000, num=n_params, dtype=np.int64),
    min_samples=linspace(2, 20, num=n_params, dtype=np.int64),
    min_micros=linspace(2, 20, num=n_params, dtype=np.int64),
    radius=linspace(0.001, 0.5, num=n_params, dtype=float, precision=precision)
)

gpreq.add_method("MCMSTStream", MCMSTStream(), param_grid=mcmc_params_grid)

# -------------------------


# ------------------------
# DenStream

beta = 0.7
denstream_params_grid = dict(
    decaying_factor=linspace(0.01, 0.1, num=n_params, dtype=float, precision=precision),
    # beta=linspace(0.01, 0.1, num=n_params, dtype=float, precision=precision),
    mu=linspace(1 / beta, 1 / beta + 100, num=n_params, dtype=float, precision=precision),
    epsilon=linspace(0.01, 0.1, num=n_params, dtype=float, precision=precision),
    n_samples_init=linspace(5, 500, num=n_params, dtype=np.int64),
    # stream_speed=linspace(100, 2000, num=n_params, dtype=np.int64),
)

gpreq.add_method("DenStream", DenStream(beta=beta), param_grid=denstream_params_grid)

## -------------------------


## ------------------------
## DBSTREAM

dbstream_params_grid = dict(
    minimum_weight=linspace(1, 3, num=n_params, dtype=float, precision=precision),
    clustering_threshold=linspace(0.05, 1, num=n_params, dtype=float, precision=precision),
    # fading_factor=linspace(0.001, 0.5, num=n_params, dtype=float, precision=precision),
    intersection_factor=linspace(0.1, 0.3, num=n_params, dtype=float, precision=precision),
    # cleanup_interval=linspace(2, 50, num=n_params, dtype=float, precision=precision)
)

gpreq.add_method("DBSTREAM", DBSTREAM(fading_factor=0.01, cleanup_interval=2), param_grid=dbstream_params_grid)

# -------------------------



### --------------------------------------
### --------------------------------------
### --------------------------------------
### Artificial data

gpreq.clear_preprocessing()

datasource = CollectionTabularData('datasets/artificial', target_names=['class', 'label', 'CLASS', 'Class'])
gpreq.set_datasource(datasource)

gpreq.add_preprocessing("norm", normalization)

gpreq.run()

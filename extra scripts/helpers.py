import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import cm
from matplotlib import colors as mcolors
from sklearn.metrics import cluster
from copy import copy


_NOISE_LABEL = -1
COLORS = ['red', 'blue', 'green', 'purple', 'yellow', 'pink', 'gray', 'darkblue', 'magenta']
MARKERS = [m for m in mpl.lines.Line2D.markers.keys() if m not in ['None', None, '', ' ']]

COLORS.extend(mcolors.TABLEAU_COLORS)


def define_matplotlib_style():
   
    plt.style.use('seaborn-v0_8-whitegrid')

    mpl.rcParams['figure.figsize'] = (10, 7)
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['image.cmap'] = 'cool'
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.titlesize'] = 'medium'
    mpl.rcParams['axes.titleweight'] = 'light'
    mpl.rcParams['axes.labelsize'] = 'medium'
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['xtick.labelsize'] = 'small'
    mpl.rcParams['ytick.labelsize'] = 'small'
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['savefig.format'] = 'svg'
    mpl.rcParams['grid.alpha'] =  0.5
    mpl.rcParams['grid.linestyle'] =  ':'
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.6
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['legend.markerscale'] = 2
    mpl.rcParams['legend.handlelength'] = 1.5
    mpl.rcParams['patch.linewidth'] = 0.5
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS) 
    

def plot_scatter2D(x1, x2, labels, s=14, xlabel=True, ylabel=True, title='', num_clusters=None, legend=True, ax=None, shuffle=True, legend_kwargs={}, random_state=1):
    
    if ax is None:
        plt.figure()
        ax = plt.axes()

    rgn = np.random.default_rng(random_state)

    ulabels = np.unique(labels) if num_clusters is None else list(range(1, num_clusters + 1))

    colors = copy(COLORS)
    markers = copy(MARKERS)
    
    if shuffle:
        rgn.shuffle(markers)
    
    for i, cid in enumerate(ulabels):

        cid = int(cid)
        marker = markers[i % len(markers)] if cid != _NOISE_LABEL else '.'
        color = colors[i % len(colors)] if cid != _NOISE_LABEL else 'k'
        label = f'C{cid}' if cid != _NOISE_LABEL else 'noise'
        ax.scatter(x1[labels==cid], x2[labels == cid], marker=marker, color=color, s=s, label=label)
      
    if xlabel:
        ax.set_xlabel('$x_1$')
    if ylabel:
        ax.set_ylabel('$x_2$')
    if legend:
        ax.legend(**legend_kwargs) 
        
    ax.set_title(title)


def plot_twolines(x1, x2, legend=True, ax=None):

    if ax is None:
        plt.figure()
        ax = plt.axes()
        
    ax.plot(x1, ls='-', label='$x_1$', color='b')
    ax.plot(x2, ls='--', label='$x_2$', color='orangered')
        
    ax.set_xlabel('Time $(t)$')
    ax.set_ylabel('Measurement')

    if legend:
        ax.legend(title='Variable')


def read_data(fname, datasetsdir='../../datasets_norm/'):
    return pd.read_csv(os.path.join(datasetsdir, fname))


def ffill(X, Y, **kwargs):
    return X.ffill(**kwargs), Y.ffill(**kwargs)


def bfill(X, Y, **kwargs):
    return X.bfill(**kwargs), Y.bfill(**kwargs)


def add_noise(X, Y=None, scale=0.001, **kwargs):
    
    noise = np.random.normal(scale=scale, size=X.shape)
    
    return X + noise, Y


def normalization(X, Y=None):
    
    Xmin = X.min()
    Xmax = X.max()
    
    X = (X - Xmin) / (Xmax - Xmin)
    
    return X, Y


def linspace(start, stop, num=100, dtype=float, precision=3):
    
    values = np.linspace(start, stop, num=num)
    values = np.round(values, decimals=precision).astype(dtype)
    values = np.unique(values)
    
    return values


def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def walk_on_directory(path, columns=None, folder_column_name='folder', file_column_name='file', 
                      file_extension='.csv'):
    
    if columns is None:
        columns = slice(None, None)
    
    if os.path.isfile(path):
         if path.endswith(file_extension):
             df = pd.read_csv(path)
             df = df[columns]
             
             path_split = path.split(os.sep)
             
             folder_name = '/'.join(path_split[1:-1])
             
             file_name = path_split[-1]
             start = file_name.find('_') + 1
             end = file_name.find('_', start)
             file_name = file_name[start:end]
             
             
             df[folder_column_name] = folder_name
             df[file_column_name] = file_name
             
             return df
         
         return pd.DataFrame()
    
    data = [pd.DataFrame()]
    for file_or_folder in os.listdir(path):
        data.append(walk_on_directory(os.path.join(path, file_or_folder),
                                      columns=columns, 
                                      folder_column_name=folder_column_name,
                                      file_column_name=file_column_name,
                                      file_extension=file_extension))
        
    return pd.concat(data, ignore_index=True)

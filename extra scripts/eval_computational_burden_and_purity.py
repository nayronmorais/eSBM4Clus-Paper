import sys
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pympler.asizeof import asizeof
from sklearn.metrics import adjusted_rand_score

from evos.clustering import *

from helpers import normalization, purity as purity_score


_MB_BASE = 1048576


def evaluate(dataset_path, dataset_name, method_name, params, compute_purity=True):
    
    class_str = f"{method_name}(**{params})".replace('SOSTream', 'SOStream')
    method = eval(class_str)
    
    data = pd.read_csv(os.path.join(dataset_path, f'{dataset_name}.csv'))
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    
    X, Y = normalization(X, Y)
    
    memory = np.zeros(shape=X.shape[0], dtype=float)
    times = np.zeros(shape=X.shape[0], dtype=float)
    num_clusters = np.zeros(shape=X.shape[0], dtype=int)
    
    for i, x in X.iterrows():
        
        x = x.values.reshape(1, -1)
        
        start_time = time.perf_counter()
        method.partial_fit(x, y=None)
        end_time = time.perf_counter()
        
        num_cluster = 0
        if hasattr(method, 'clusters_'):
            num_cluster = len(method.clusters_)
            
        elif  hasattr(method, 'micro_clusters_'):
            num_cluster = len(method.micro_clusters_)
            
        elif  hasattr(method, 'micro_clusters'):
            num_cluster = len(method.micro_clusters)
            
        elif hasattr(method, 'clusters'):
            num_cluster = len(method.clusters)
        
        memory[i] = asizeof(method) / _MB_BASE
        times[i] = end_time - start_time
        num_clusters[i] = num_cluster
        
    purity_value = 0.0
    if compute_purity:
        Ypred = method.predict(X)
        purity_value = purity_score(Y, Ypred)
    
    return memory, times, num_clusters, purity_value
        

if __name__ == '__main__':
    
    force_run = True
    compute_purity = True
    
    save_runs_file = False
    save_purity_file = True
    
    num_runs = 1
    
    datasets_path = '../datasets/'
    save_path = '../formatted results/computational burden'
    
    os.makedirs(save_path, exist_ok=True)
    
    search_results_file_name = '../formatted results/search_results.csv'
    
    search_result = pd.read_csv(search_results_file_name, index_col=['Dataset', 'Method'])

    target_datasets = ['dim32', 'dim64', 'dim128', 'dim256', 'dim512', 'dim1024']
    computational_result = pd.DataFrame(index=search_result.index)
    
    datasets_name = ['artificial/s1', 'artificial/s2', 'artificial/s3', 'artificial/s4',
                    'artificial/cassini', 'artificial/chameleon-ds3-clean', 'artificial/chameleon-ds4-clean',
                    'artificial/chameleon-ds3-with-noise',
                    'artificial/chameleon-ds4-with-noise',
                    'artificial/dim32', 'artificial/dim64', 'artificial/dim128', 'artificial/dim256',
                    'artificial/dim512', 'artificial/dim1024']
    
    methods_name_table = ['DenStream', 'DBSTREAM', 'CEDAS',  'SOSTream', 'MacroSOStream',
                          'MCMSTStream', 'MicroTEDAClus', 'AutoCloud', 'OEC', 'eSBM4Clus']
    
    
    purity_table = pd.DataFrame(index=datasets_name, columns=methods_name_table, dtype=float)
    
    for (dataset, method_name), row in search_result.iterrows():
        
        
        
        dataset_path, dataset_name = dataset.split('/')
        dataset_path = os.path.join(datasets_path, dataset_path)
            
        params = row['params']
        
        name = f'{dataset_name}_{method_name}'      
        computational_file_name = f'{save_path}/{name}.csv'
        
        if not os.path.exists(computational_file_name) or force_run:
                    
            computationalb_df = pd.DataFrame()
            
            for run in range(num_runs):
                
                print(f"Processing dataset {dataset} with method {method_name} run {run+1}/{num_runs}")
                
                memory, times, num_clusters, purity_value  = evaluate(dataset_path, dataset_name, method_name, params, compute_purity)
            
                computationalb_df[f'memory_r{run}'] = memory
                computationalb_df[f'time_r{run}'] = times
                computationalb_df['clusters'] = num_clusters
            
            if save_runs_file:
                computationalb_df.to_csv(computational_file_name, index=False)
            
            purity_table.loc[dataset, method_name] = purity_value
            
        else:
            
           print(f'[INFO] Loading the result ({computational_file_name}) of the last the run, to force a run set `force_run=True`.')
           computationalb_df = pd.read_csv(computational_file_name) 
        
    
    if compute_purity:
        datasets_name_table = ['S1', 'S2', 'S3', 'S4',
                               'Ca', 'DS3', 'DS4', 'DS3wN', 'DS4wN', 
                               'DIM32', 'DIM64', 'DIM128', 'DIM256','DIM512', 'DIM1024'] 
        
        purity_table.index = datasets_name_table
        
        if save_purity_file:
            purity_table.to_csv('../formatted results/purity_results.csv')
        
        purity_table.loc['Mean', :] = purity_table.mean()
        purity_table = purity_table.sort_values(by='Mean', axis=1, ascending=False)
        print(purity_table.to_latex(float_format='%.4f'))
        
        

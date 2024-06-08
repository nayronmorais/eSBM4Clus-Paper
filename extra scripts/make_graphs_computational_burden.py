import sys

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pympler.asizeof import asizeof
from sklearn.metrics import adjusted_rand_score

from helpers import normalization, purity as purity_score, define_matplotlib_style

from collections import defaultdict



if __name__ == '__main__':
    
    define_matplotlib_style()
    
    datasets_path = '../datasets/'
    load_path = '../formatted results/computational burden'
    save_path = '../formatted results/computational burden/graphs'
    
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
    
    methods_names = ['DenStream', 'DBSTREAM', 'CEDAS',  'SOSTream', 'MacroSOStream',
                          'MCMSTStream', 'MicroTEDAClus', 'AutoCloud', 'OEC', 'eSBM4Clus']
    
    datasets_name_ids = ['S1', 'S2', 'S3', 'S4',
                           'Ca', 'DS3', 'DS4', 'DS3wN', 'DS4wN', 
                           'DIM32', 'DIM64', 'DIM128', 'DIM256','DIM512', 'DIM1024'] 
    
    
    for dataset, dataset_id in zip(datasets_name, datasets_name_ids):
        
        
        dims_dataset = False
        
        figure_memory, ax_memory = plt.subplots(1, 1, figsize=(6, 4))
        figure_time, ax_time = plt.subplots(1, 1, figsize=(6, 4))
        figure_clusters, ax_clusters = plt.subplots(1, 1, figsize=(6, 4))
        
        ax_memory.set_xlabel("Sample (t)")
        ax_time.set_xlabel("Sample (t)")
        ax_clusters.set_xlabel("Sample (t)")
        ax_memory.set_ylabel("Memory (MB)")
        ax_time.set_ylabel("Time (ms)")
        ax_clusters.set_ylabel("Num. of clusters")
        
        
      
        
        dataset_path, dataset_name = dataset.split('/')
        dataset_path = os.path.join(datasets_path, dataset_path)
        
        if dataset_id.startswith("DIM"):
            ax_time_denstream = ax_time.twinx()
            ax_time_denstream.spines['right'].set_visible(True)
        
        for method_name in methods_names:

            print(f"Processing dataset {dataset} with method {method_name}:")
            
            name = f'{dataset_name}_{method_name}'      
            computational_file_name = f'{load_path}/{name}.csv'
        
            computational_df = pd.read_csv(computational_file_name)
            
            memory = computational_df['memory_r0'].values
            time = computational_df['time_r0'].values
            clusters = computational_df['clusters']
            
            num_runs = 1
            for column_name in computational_df.columns:
                
                if column_name.startswith('memory') and column_name != 'memory_r0':
                    memory += computational_df[column_name].values
                    
                elif column_name.startswith('time') and column_name != 'time_r0':
                    time += computational_df[column_name].values
                 
                if not column_name.endswith('r0'):
                    num_runs += 1
                
            memory /= num_runs
            time /= num_runs
            
            time *= 1000 # seconds to milliseconds
            
            ax_memory.plot(memory, label=method_name)
            ax_clusters.plot(clusters, label=method_name)
            
            if method_name == 'DenStream' and dataset_id.startswith("DIM"):
                ln, = ax_time_denstream.plot(time, label=method_name)
                color = ln.get_color()
                
                ax_time_denstream.tick_params(axis='y', colors=color)
                ax_time_denstream.yaxis.label.set_color(color)
                ax_time_denstream.spines['right'].set_color(color)
            
            else:
                ax_time.plot(time, label=method_name)
            
        ax_memory.legend(loc='upper left')
        ax_time.legend(loc='upper left')
        ax_clusters.legend(loc='upper left')
        
        figure_memory.savefig(f'{save_path}/{dataset_name}_memory.pdf')
        figure_time.savefig(f'{save_path}/{dataset_name}_time.pdf')
        figure_clusters.savefig(f'{save_path}/{dataset_name}_number_of_clusters.pdf')
        

        
        

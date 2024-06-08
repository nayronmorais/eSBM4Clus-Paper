import os
import numpy as np
import pandas as pd

from helpers import walk_on_directory


COLUMNS = ['rank_test_score', 'split0_test_score', 'mean_test_score', 'params']
      
        
if __name__ == '__main__':
    
    precision = 4
    results_path = '../results'    
    results = walk_on_directory(results_path, 
                                columns=COLUMNS,
                                folder_column_name='Dataset',
                                file_column_name='Method')
    
    summarized_results = results.where(results['rank_test_score'] == 1, inplace=False).dropna(axis=0)
    summarized_results = summarized_results.drop_duplicates(subset=['Dataset', 'Method'])
    
    datasets = summarized_results['Dataset']
    methods = summarized_results['Method']
    
  
    values = [datasets, methods]
    multindex = pd.MultiIndex.from_arrays(values, names=('Dataset', 'Method'))
    
    summarized_results = summarized_results.drop(labels=['Dataset', 'Method'], axis=1)
    summarized_results = summarized_results.set_index(multindex)
    
    summarized_results.to_csv('../formatted results/search_results.csv')
    
    
    ### ----------------------------------------------------------
    ### ---------------------------------------------------------
    ### For latex table view
    # datasets_name = np.unique(datasets)
    # methods_name = sorted(np.unique(methods))
    
    datasets_name = ['artificial/s1', 'artificial/s2', 'artificial/s3', 'artificial/s4',
                    'artificial/cassini', 'artificial/chameleon-ds3-clean', 'artificial/chameleon-ds4-clean',
                    'artificial/chameleon-ds3-with-noise',
                    'artificial/chameleon-ds4-with-noise',
                    'artificial/dim32', 'artificial/dim64', 'artificial/dim128', 'artificial/dim256',
                    'artificial/dim512', 'artificial/dim1024']
    
    methods_name = ['DenStream', 'DBSTREAM', 'CEDAS',  'SOSTream', 'MacroSOStream',
                    'MCMSTStream', 'MicroTEDAClus', 'AutoCloud', 'OEC', 'eSBM4Clus']
    
    table_df = pd.DataFrame(index=datasets_name, columns=methods_name, dtype=float)#columns=datasets_name, index=methods_name)
    for (dataset_name, method_name), row in summarized_results.iterrows():
      
        table_df.loc[dataset_name, method_name] = row['split0_test_score']
    
    datasets_name_table = ['S1', 'S2', 'S3', 'S4',
                           'Ca', 'DS3', 'DS4', 'DS3wN', 'DS4wN', 
                           'DIM32', 'DIM64', 'DIM128', 'DIM256','DIM512', 'DIM1024']
    
    table_df.index = datasets_name_table
    table_df.loc['Mean', :] = table_df.mean()
    table_df = table_df.sort_values(by='Mean', axis=1, ascending=False)
    print(table_df.to_latex(float_format='%.4f'))

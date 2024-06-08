import os
import pandas as pd
import numpy as np

import plotly
import plotly.express as px
import plotly.io as pio

from helpers import walk_on_directory


# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
COLUMNS = ['split0_test_score', 'mean_test_score', 'params']
      
        
if __name__ == '__main__':
    
    precision = 4
    results_path = '../results'   
    save_path = '../formatted results/varying hyperparameters/'
    
    os.makedirs(save_path, exist_ok=True)
    
    results = walk_on_directory(results_path, 
                                columns=COLUMNS,
                                folder_column_name='Dataset',
                                file_column_name='Method')
    
    datasets_name = ['artificial/s1', 'artificial/s2', 'artificial/s3', 'artificial/s4',
                    'artificial/cassini', 'artificial/chameleon-ds3-clean', 'artificial/chameleon-ds4-clean',
                    'artificial/chameleon-ds3-with-noise',
                    'artificial/chameleon-ds4-with-noise',
                    'artificial/dim32', 'artificial/dim64', 'artificial/dim128', 'artificial/dim256',
                    'artificial/dim512', 'artificial/dim1024']
    
    datasets_name_table = ['S1', 'S2', 'S3', 'S4',
                           'Ca', 'DS3', 'DS4', 'DS3wN', 'DS4wN', 
                           'DIM32', 'DIM64', 'DIM128', 'DIM256','DIM512', 'DIM1024']
    
    methods_name = ['DenStream', 'DBSTREAM', 'CEDAS',  'SOSTream', 'MacroSOStream',
                    'MCMSTStream', 'MicroTEDAClus', 'AutoCloud', 'OEC', 'eSBM4Clus']
    
    table_link = pd.DataFrame(index=datasets_name, columns=methods_name, dtype=str)
    link = 'https://github.com/nayronmorais/eSBM4Clus-paper/tree/main/'
    
    
    for ((dataset, method), subresult) in results.groupby(['Dataset', 'Method']):
        
        print(f"{dataset} - {method}")
        
        dataset_path, dataset_name = dataset.split('/')
        
        params = subresult['params']
        params = pd.DataFrame(map(eval, params))
        params['ARI'] = subresult[COLUMNS[0]].values
        
        hyperparameters = params.columns[:-1]
        
        labels  = {}
        # if method == 'eSBM4Clus':
        #     labels = {'beta': '$\beta$'}
        
        
        fig_fname = f"{save_path}hyper_{dataset_name}_{method}.pdf"
      
        fig = px.parallel_categories(params, dimensions=hyperparameters,
                        labels=labels, color='ARI', 
                        color_continuous_scale=px.colors.sequential.Magma)
        
        
        margin={'t': 17,'l':14,'b':0.1,'r': 0.1}
        
        width = 900 
        if len(hyperparameters) == 1:
            margin['l'] = 0.01
            width = 200
        
        fig.update_layout(
            margin=margin
        )
        fig.write_image(fig_fname, scale=0.1, width=width)
        # fig.show()
    
        table_link.loc[dataset, method] = ('\href{%s/%s}{open fig}' % (link, fig_fname)).replace('_', '\_')
        
        
    table_link.index = datasets_name_table
    
    print(table_link.to_latex(escape=False))
    

import os
import pandas as pd


def read_norm_save(fpath, newfpath):
    
    data = pd.read_csv(fpath)
    
    X = data.iloc[:, :-1]
    
    min_ = X.min()
    max_ = X.max()
    
    Xnorm = (X - min_) / (max_ - min_)
    
    data.iloc[:, :-1] = Xnorm
    
    data.to_csv(newfpath, index=False)
    

if __name__ == '__main__':
    
    rootdir = '../datasets'
    new_rootdir = rootdir + '_norm'
    
    os.makedirs(new_rootdir, exist_ok=True)
    
    for file_or_dir in os.listdir(rootdir):
        
        path = os.path.join(rootdir, file_or_dir)
       
        if path.endswith('.csv'):
            newfpath = os.path.join(new_rootdir, file_or_dir)
            read_norm_save(path, newfpath)
            
        else:
          
            for file_or_dir_2 in os.listdir(path):
                path2 = os.path.join(path, file_or_dir_2)
                if path2.endswith('.csv'):
                    newfpath = os.path.join(new_rootdir, file_or_dir_2)
                    read_norm_save(path2, newfpath)
        
        
        
        

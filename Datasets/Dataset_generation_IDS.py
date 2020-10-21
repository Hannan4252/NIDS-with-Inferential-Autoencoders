import pandas as pd
import numpy as np
import os

'''
This scripts read all of the csv files and generate a single hd5 file for all of
these datasets and then create a subset of 10,5 and 2.5
prercents to train and test different tehniques


Probability distribution is kept same for all attacks and normal traffic 
'''

df = pd.DataFrame()

##Reading files from 1st day
path, _, file_names =next(os.walk('/mnt/datasets/novelty/cic-ids-2017/205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCVE/'))
for each_name in file_names:
    file_path = path +each_name
    if file_path.endswith('.csv'):
        print(file_path)
        tmp = pd.read_csv(file_path)
        tmp = tmp[~tmp.isin([np.nan, np.inf, -np.inf]).any(1)]
        tmp = tmp.drop(columns=  [' Destination Port'])
        #print(df.columns)
        df = df.append(tmp, ignore_index=True)

##write whole of the dataset to a single file
print(df.shape)        
df.to_hdf('./Datasets/CIC_IDS_2017/ids_2017.h5', key='ids_2017')

unique_Labels = np.unique(df[' Label'])


##writting 50 percent
subset =pd.DataFrame()
for each_unqiue_label in unique_Labels:
    subset = subset.append(df[df[' Label'] ==each_unqiue_label].sample(frac =0.5 , random_state = 0xFFFF))
    
print(subset.shape)
subset.to_hdf('./Datasets/CIC_IDS_2017/ids_2017_50.h5', key='ids_2017_50')

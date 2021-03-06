#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:38:56 2022

@author: deeplearn
"""
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

train_path = 'ISIC_2019_Training_Input/ISIC_2019_Training_Input'
move_path = 'train/'
csv_file_train = pd.read_csv('ISIC_2019_Training_GroundTruth.csv', delimiter=',')
class_names = list(csv_file_train.columns[1:])


for folder_name in class_names:
    if not os.path.exists(os.path.join(move_path, folder_name)):
        os.makedirs(os.path.join(move_path, folder_name))
    
datas = np.array(csv_file_train)
images = np.array(datas[:,0], dtype=str)
folders = np.array(datas[:,1:], dtype=int)

#%%
for i in tqdm(range(len(folders))):
    for j in range(len(class_names)):
        if folders[i,j] == 1:
            src = os.path.join(train_path, images[i]+'.jpg')
            dest = os.path.join(move_path, class_names[j], images[i]+'.jpg')
            shutil.copy2(src, dest) 
            
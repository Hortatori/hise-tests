import pandas as pd
import numpy as np
from os.path import exists


df_np = np.load('raw_data/Event2018/french_tweets.npy', allow_pickle=True)

splitted_np = np.array_split(df_np,4)

folder = "data/test_set/"
for i in range(len(splitted_np)) :
    df_np_path = folder + str(i) + '.npy'
    print(df_np_path)
    if not exists(df_np_path):
        # np.save(df_np_path, splitted_np[i] )
        print(i, type(splitted_np[i]))
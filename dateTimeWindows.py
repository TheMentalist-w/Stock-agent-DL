import warnings
import pandas as pd
import glob
import numpy as np

def load_matrix(path=""):
    testing=False
    data = pd.read_csv('matrix.csv')
    data_diff=data.loc[:, data.columns!='Date']
    resources_names=list(data_diff.columns)
    for i in resources_names:
        for j in resources_names:
            if i==j:
                break
            compare_two=pd.DataFrame(data={i:data[i], j: data[j]})
            print(compare_two)


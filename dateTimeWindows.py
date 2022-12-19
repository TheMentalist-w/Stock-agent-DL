import warnings
import pandas as pd
import glob
import numpy as np

def load_matrix(path=""):
    testing=False
    data = pd.read_csv('matrix.csv')
    compare_matrix=pd.DataFrame(data=data['Date'])
    data_diff=data.loc[:, data.columns!='Date']
    resources_names=list(data_diff.columns)
    sum_days=5 #specifies length of time window
    for i in resources_names:
        for j in resources_names:
            if i==j:
                break
            compare_two=pd.DataFrame(data={i:data[i], j: data[j]})
            c2=compare_two.pct_change()
            compare_two[f"{i}vs{j}"]=np.std([c2[i],c2[j]],axis=0)/(c2[i]+c2[j]) #comparison
            compare_two=compare_two[f"{i}vs{j}"].rolling(sum_days).sum()

            if testing:print(compare_two)
            compare_matrix[f"{i}vs{j}"]=compare_two
    if testing: print(compare_matrix)
    compare_matrix=compare_matrix.iloc[sum_days:]
    compare_matrix.to_csv('comparison_matrix.csv')
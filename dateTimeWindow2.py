import warnings
from collections import defaultdict, Counter
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

    long_list=[]
    for i in resources_names:
        for j in resources_names:
            if i==j:
                break
            compare_two=pd.DataFrame(data={i:data[i], j: data[j]})
            c2=compare_two.diff(axis=1)
            c2=c2[j].pct_change()
            long_list.append([round(c,4) for c in c2[1:]])
    tokens=[]
    for l in long_list:
        for w in range(len(l)-sum_days+1):
            tokens.append(str(l[w:w+sum_days]))
    unique_tokens=np.unique(tokens)
    token_dict=dict.fromkeys(unique_tokens, 0)
    print(token_dict)
    for w in tokens:
        token_dict[w]+=1
    if testing: print(w2i)
    df = pd.DataFrame(list(token_dict.items()), columns=['trend', 'number']) #TODO - check for the correct word, eluded me rn
    df.to_csv('tokens.csv')
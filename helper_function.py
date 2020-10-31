import numpy as np
import sklearn
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def savenp(inp_file, out_dir):
    pass

def shuffle(inp_file, out_file, test_size, ret = False):
    np_obj = np.load(inp_file)
    inds_split = train_test_split(range(np_obj["features"].shape[0]), np_obj["labels"], 
    test_size=test_size)
    obj = dict()
    inds_shuff = inds_split[0] + inds_split[1]
    obj["features"] =  np_obj["features"][inds_shuff]
    obj["labels"] =  np_obj["labels"][inds_shuff]
    obj["accessions"] =  np_obj["accessions"][inds_shuff]
    np.savez(out_file, **obj)
    # with open(out_file, "wb") as f:
    #     np.savez(f, obj["features"])
    #     np.savez(f, obj["labels"])
    #     np.savez(f, obj["accessions"])
    if(ret):
        return(inds_shuff)

def table(v1, v2 = None):
    if(v2 is None):
        return pd.Series(v1).value_counts()
    else:
        return pd.Series([(i,j) for i,j in zip(v1, v2)]).value_counts()

def createDf(npOb):
    labs = npOb["labels"]
    accs = npOb["accessions"]
    df = pd.DataFrame({"labs":npOb["labels"], "accs":npOb["accessions"])
    return df
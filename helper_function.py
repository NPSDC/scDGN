import numpy as np
import sklearn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from utils.vis_util import extract_rep
from utils.train_util import ClassicTrainer, ADGTrainer
from utils.data_util import dataset, dataset_CADG

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
    df = pd.DataFrame({"labs":npOb["labels"], "accs":npOb["accessions"]})
    return df

def get_best_mod(pi_file):
    df = pickle.load(open(pi_file, "rb"))
    values = list(df.values())
    lam_max = list(df.keys())[np.argmax([max(list(d.values())) for d in list(df.values())])]
    margin_max = list(df[lam_max].keys())[np.argmax(df[lam_max].values())]
    return lam_max, margin_max

def conv_ad_data(file_pan, file_pbmc, orig_folder, save_fold):
    data_pan = np.load(file_pan)
    data_pbmc = np.load(file_pbmc)
    pan_ds = [0,1,2,3,4,5]
    for i in pan_ds:
        pan = dict()
        pan_orig = np.load(os.path.join(orig_folder, "pancreas" + str(i) + ".npz"))
        pan["features"] = np.concatenate((data_pan["training"+str(i)], data_pan["test"+str(i)]))
        pan["labels"] = pan_orig["labels"]
        pan["accessions"] = pan_orig["accessions"]
        np.savez(os.path.join(save_fold, "pancreas" + str(i) + ".npz"), **pan)
    
    pbmc = dict()
    pbmc_orig = np.load(os.path.join(orig_folder, "pbmc.npz"))
    pbmc["features"] = np.concatenate((data_pbmc["training"], data_pbmc["test0"]))
    pbmc["labels"] = pbmc_orig["labels"]
    pbmc["accessions"] = pbmc_orig["accessions"]
    np.savez(os.path.join(save_fold, "pbmc.npz"), **pbmc)

def ret_test_acc(dir, end = -2):
    acc = {"pancreas0":0, "pancreas1":0, "pancreas2":0, "pancreas3":0, "pancreas4":0, "pancreas5":0, "pbmc":0}
    pan_dirs = sorted(os.listdir(dir))[0:7]
    acc_keys = list(acc.keys())
    for i in range(len(pan_dirs)):
        d = pan_dirs[i]
        if acc_keys[i] in d:
            df_dir = os.path.join(dir, d)
            files = os.listdir(df_dir)
            files = [f for f in files if f.endswith("txt")]
            with open(os.path.join(df_dir, files[0]), "rb") as f:
                #print(str(f.readlines()[-2]).split(" "))
                acc[acc_keys[i]] = round(float(str(f.readlines()[-2]).split(" ")[end].split(",")[0]),3)
        else:
            print("not key")
    return acc

def gen_npz(data_dir, mod_dir, inp_dim = 3000, batch_size = 256, validation = False, dim1 = 1136, 
dim2 = 100, dom_dim = 64, n_epoch = 250):
    acc = {"pancreas0":0, "pancreas1":0, "pancreas2":0, "pancreas3":0, "pancreas4":0, "pancreas5":0, "pbmc":0}
    pan_dirs = sorted(os.listdir(mod_dir))
    pan_dirs = [pan for pan in pan_dirs if pan.startswith("pbmc") or pan.startswith("pancreas")]
    acc_keys = list(acc.keys())

    for i in range(len(pan_dirs)):
        d = pan_dirs[i]
        if acc_keys[i] in d:
            df_dir = os.path.join(mod_dir, d)
            files = os.listdir(df_dir)
            files = [f for f in files if f.startswith("best_model")]
            mod_path = os.path.join(mod_dir, pan_dirs[i])
            label_size=13
            if("pbmc" in acc_keys[i]):
                label_size = 10
            if("scDGN" in mod_dir):
                data = dataset_CADG(data_dir, batch_size, label_size = label_size, dataset_name = acc_keys[i], validation = validation)
                t = ADGTrainer(inp_dim, 1, 1, dim1, dim2, label_size, dom_dim, n_epoch, batch_size, mod_path, validation=validation)
            else:
                data = dataset(data_dir, batch_size, label_size = label_size, dataset_name = acc_keys[i], validation = validation)
                t = ClassicTrainer(inp_dim, dim1, dim2, label_size, n_epoch, batch_size, mod_path, validation=validation, use_gpu=1)
                
            t.dataset = data
            t.load_model(os.path.join(mod_path, files[0]))
            if("scDGN" in mod_dir):
                representations, labels, domains = extract_rep(t, data, type="concat", method = "scDGN")
            else:
                representations, labels, domains = extract_rep(t, data, type="concat", method = "NN")
            df = {}
            df["features"] = representations
            df["labels"] = labels
            df["domains"] = domains
            
            np.savez(os.path.join(mod_dir, d, acc_keys[i] + ".npz"), **df)
import os
import argparse
import numpy as np
from utils.train_util import ClassicTrainer, ADGTrainer
from utils.data_util import dataset, dataset_CADG
import pickle

N_CELL_TYPES = {'scquery':39, 'pbmc':10,  'pancreas0': 13, 'pancreas1': 13, 'pancreas2': 13, 'pancreas3': 13, 
                'pancreas4': 13, 'pancreas5': 13}

N_GENES = {'scquery':20499, 'pbmc':3000, 'pancreas0': 3000,  'pancreas1': 3000, 'pancreas2': 3000, 'pancreas3': 3000, 
                'pancreas4': 3000, 'pancreas5': 3000}

def train(args, dim_i, dim_o, data_path):
    model_path = os.path.join(args.ckpts, args.output)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if args.adv_flag:
        dataloader = dataset_CADG(data_path, args.batch_size, label_size=dim_o, dataset_name=args.dataset, validation=args.dataset)
        trainer = ADGTrainer(dim_i, args.margin, args.lamb, args.dim1, args.dim2, dim_o, args.dimd, args.epochs, 
                        args.batch_size, model_path, use_gpu=args.use_gpu, validation=args.validation, save_mod = args.save_mod)
        log_file = '%s_scDGN_margin%.2f_lambda%.2f_%s.txt'%(args.dataset, args.margin, args.lamb, args.output)
    else:
        dataloader = dataset(data_path, args.batch_size, label_size=dim_o, dataset_name=args.dataset, validation=args.dataset)
        trainer = ClassicTrainer(dim_i, args.dim1, args.dim2, dim_o, args.epochs, args.batch_size, 
                            model_path, use_gpu=args.use_gpu, validation=args.validation, save_mod = args.save_mod)
        log_file = '%s_NN_%s.txt'%(args.dataset, args.output)
    trainer.dataset = dataloader
    
    with open(os.path.join(args.ckpts, args.output, log_file), 'w') as fw:
        fw.write(str(args)+'\n')
        trainer.train(fw)
    if(args.validation):
        return trainer.best_valid_acc

def valid_scdgn(lambdas, margin, args, n_genes, n_labels):
    valid_acc = dict()
    out_file = os.path.join(args.ckpts, args.output, "valid_acc.pi")
    if(os.path.exists(out_file)):
        with open(out_file, "rb") as f:
            valid_acc = pickle.load(f)
    

    for l in lambdas:
        if not l in valid_acc.keys():
            valid_acc[l] = {}
        for m in margin:
            args.lamb = l
            args.margin = m
            valid_acc[l][m]=train(args, n_genes, n_labels, args.inp_folder)
        with open(out_file, "wb") as f:
            pickle.dump(valid_acc, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpts', type=str, default='ckpts/', help='model checkpoints path')

    parser.add_argument('-d1', '--dim1', type=int, default=1136, help='number of hidden units in layer1')
    parser.add_argument('-d2', '--dim2', type=int, default=100, help='number of hidden units in layer2')
    parser.add_argument('-dd', '--dimd', type=int, default=64, help='number of hidden units in domain discriminator')
    parser.add_argument('-ng', '--ngenes', type=int, default=3000, help='number of genes in the input')

    parser.add_argument("-dn", "--dataset", type=str, default='scquery', help='name of dataset')
    parser.add_argument('--validation', type=int, default=1, help='using validation set or not')

    parser.add_argument("-o", "--output", type=str, default='scquery', help='Save model filepath')
    parser.add_argument('-e', '--epochs', type=int, default=250, help='number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-i', '--inp_folder', type=str, default="scDGN", help='Input folder to take data from')
    parser.add_argument('--save_model', type=int, default=1, help='save the trained model?')
    
    parser.add_argument('-adv', '--adv_flag', type=int, default=1, help='conditional domain adversarial training')
    parser.add_argument('-l', '--lamb', type=float, default=1.0, help='trade-off between domain invariance and classification')
    parser.add_argument('-m', '--margin', type=float, default=1.0, help='margin of contrastive loss')
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu to train the model')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for training')
    parser.add_argument('-s', '--save_mod', type=int, default=1, help='saving model or not')
    parser.add_argument('-v', '--run_val', type=int, default=0, help='Run validation for hyperparameter tuning or not')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.dataset not in N_CELL_TYPES.keys():
        print('Dataset not implemented!')

    if not os.path.exists(args.ckpts):
        os.mkdir(args.ckpts)

    n_labels = N_CELL_TYPES[args.dataset]
    n_genes = N_GENES[args.dataset]
    if(args.ngenes != 3000):
        n_genes = args.ngenes
    data_path = os.path.join('data_shuff',args.inp_folder)
    #args.epochs = 100
    if args.run_val == 1: 
        if args.adv_flag == 1:
            valid_scdgn(np.arange(0.50, 1.05, 0.05), np.arange(1,10,1), args, n_genes, n_labels)
    else:
        train(args, n_genes, n_labels, args.inp_folder)
    

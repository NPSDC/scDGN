#!/bin/bash
#SBATCH -J run_scdgn # Job name
#SBATCH --mail-user=npsingh@umd.edu # Email for job info
#SBATCH --mail-type=fail,end# Get email for begin, end, and fail
#SBATCH --time=0-4:00:00
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:1

source /fs/classhomes/fall2020/cmsc828w/c828w073/miniconda3/bin/activate
conda activate scDGN

python run_scDGN.py -dn pancreas0 -g 0 -adv 0 -o pancreas0_test_ADAE100 --validation 1 -i data_shuff/ADAE_100 -c ckpts/ADAE_100 -ng 100

python run_scDGN.py -dn pancreas1 -g 0 -adv 0 -o pancreas1_test_ADAE100 --validation 1 -i data_shuff/ADAE_100 -c ckpts/ADAE_100 -ng 100

python run_scDGN.py -dn pancreas2 -g 0 -adv 0 -o pancreas2_test_ADAE100 --validation 1 -i data_shuff/ADAE_100 -c ckpts/ADAE_100 -ng 100

python run_scDGN.py -dn pancreas3 -g 0 -adv 0 -o pancreas3_test_ADAE100 --validation 1 -i data_shuff/ADAE_100 -c ckpts/ADAE_100 -ng 100

python run_scDGN.py -dn pancreas4 -g 0 -adv 0 -o pancreas4_test_ADAE100 --validation 1 -i data_shuff/ADAE_100 -c ckpts/ADAE_100 -ng 100

python run_scDGN.py -dn pancreas5 -g 0 -adv 0 -o pancreas5_test_ADAE100 --validation 1 -i data_shuff/ADAE_100 -c ckpts/ADAE_100 -ng 100

python run_scDGN.py -dn pbmc -g 0 -adv 0 -o pbmc_test_ADAE100 --validation 1 -i data_shuff/ADAE_100 -c ckpts/ADAE_100 -ng 100




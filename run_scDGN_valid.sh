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
# python run_scDGN.py -dn pancreas0 -l 0.1 -m 1 -g 0 -adv 1 -o pancreas0_valid_CDGN --validation 1 -i data_shuff/scDGN/ -c ckpts/scDGN -s 0 -v 1

# python run_scDGN.py -dn pancreas1 -l 0.1 -m 1 -g 0 -adv 1 -o pancreas1_valid_CDGN --validation 1 -i data_shuff/scDGN/ -c ckpts/scDGN -s 0 -v 1

# python run_scDGN.py -dn pancreas2 -l 0.1 -m 1 -g 0 -adv 1 -o pancreas2_valid_CDGN --validation 1 -i data_shuff/scDGN/ -c ckpts/scDGN -s 0 -v 1

# python run_scDGN.py -dn pancreas3 -l 0.1 -m 1 -g 0 -adv 1 -o pancreas3_valid_CDGN --validation 1 -i data_shuff/scDGN/ -c ckpts/scDGN -s 0 -v 1

# python run_scDGN.py -dn pancreas4 -l 0.1 -m 1 -g 0 -adv 1 -o pancreas4_valid_CDGN --validation 1 -i data_shuff/scDGN/ -c ckpts/scDGN -s 0 -v 1

# python run_scDGN.py -dn pancreas5 -l 0.1 -m 1 -g 0 -adv 1 -o pancreas5_valid_CDGN --validation 1 -i data_shuff/scDGN/ -c ckpts/scDGN -s 0 -v 1

# python run_scDGN.py -dn pbmc -l 0.1 -m 1 -g 0 -adv 1 -o pbmc_valid_CDGN --validation 1 -i data_shuff/scDGN/ -c ckpts/scDGN -s 0 -v 1




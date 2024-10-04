#!/bin/bash -l

#SBATCH --partition=volta-x86
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32768
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --job-name=train14
#SBATCH --output=job_%x.log
#SBATCH --error=job_%x.err
#SBATCH --mail-user=rke@lanl.gov
#SBATCH --mail-type=ALL

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $SLURM_JOB_NAME"
echo "Job ID : $SLURM_JOB_ID" 
echo "=========================================================="
cat /etc/redhat-release

MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MASTERPORT=6000

module purge
module load cuda/11.3.1
module load miniconda3/py39_4.10.3
source activate covidtransformer

export OMP_NUM_THREADS=1
#Specify your own working directory where the testing codebase is located.
cd ~/CovidNet/src

srun python -u train_new_noise_smooth.py \
  --used_model transformer_encoder2 -lr 0.001 -wd 0.05 --batch_size 256 \
  --num_epochs 1000 --lr_scheduler --dropout 0. --norm\
  --day 42 --future 14 \
  --ckpt_name final_drop_token \
  --ckpt_dir ~/CovidNet/ckpt --anno_path ~/CovidNet/dataset \

# change --future to train models for different days prediction.
# use seed_list to indicate the random seed, default for 0-15
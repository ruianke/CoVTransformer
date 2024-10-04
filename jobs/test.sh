#!/bin/bash -l

#SBATCH --partition=volta-x86
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32768
#SBATCH --qos=long
#SBATCH --time=12:00:00
#SBATCH --job-name=test
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

# The models' checkpoints are hard code on test_save_final.py line 649
srun python -u test_save_final.py \
 --batch_size 256 --used_model transformer_encoder2 --day 42 \
 --ckpt_name test \
 --dataset_version v4 \
 --ckpt_dir ~/CovidNet/ckpt --anno_path ~/CovidNet/dataset \
 --output_dir ~/CovidNet/result --save_file_name all_results_best_14_21_28_35_42_60_latest3_noise


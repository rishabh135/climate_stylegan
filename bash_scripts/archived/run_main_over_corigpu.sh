#!/bin/bash
#SBATCH -J vtj 
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --mail-user=rgupta2@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH --comment=96:00:00
#SBATCH --time-min=16:00:00 #the minimum amount of time the job should run
#SBATCH --time=24:00:00
#SBATCH --error=vtj-%j.err
#SBATCH --output=vtj-%j.out

#SBATCH --signal=B:USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

# use the following three variables to specify the time limit per job (max_timelimit), 
# the amount of time (in seconds) needed for checkpointing, 
# and the command to use to do the checkpointing if any (leave blank if none)
max_timelimit=24:00:00   # can match the #SBATCH --time option but don't have to
ckpt_overhead=60         # should match the time in the #SBATCH --signal option
ckpt_command=

# requeueing the job if reamining time >0 (do not change the following 3 lines )
. /usr/common/software/variable-time-job/setup.sh
requeue_job func_trap USR1
#

# # user setting goes here
# export OMP_PROC_BIND=true
# export OMP_PLACES=threads
# export OMP_NUM_THREADS=8


#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


module load cuda
module load esslurm
module load python3
module load tensorflow 


#run the application:
python /global/homes/r/rgupta2/rgupta2/StyleGAN/src/StyleGAN-Tensorflow/main.py 


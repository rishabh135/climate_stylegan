#!/bin/sh

module load esslurm
module load cuda/10.0.130

module load python/3.7-anaconda-2019.07
source activate tf-1.13
module load tensorflow/gpu-1.13.1-py36

python -c "import tensorflow as tf; print(tf.__version__); print(tf.__path__)"


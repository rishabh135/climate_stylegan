for i in {0..1};do sbatch --dependency=singleton --job-name=StyleGAN train_sbatch ; done

for i in {0..5};
do sbatch --dependency=singleton  $1; 
done
for i in {0..6};
do sbatch --dependency=singleton  $1; 
done
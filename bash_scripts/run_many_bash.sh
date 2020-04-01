for i in {0..18};
do sbatch --dependency=singleton  $1; 
done
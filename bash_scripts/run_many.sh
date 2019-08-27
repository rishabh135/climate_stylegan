for i in {0..8};
do sbatch --dependency=singleton  $1; 
done

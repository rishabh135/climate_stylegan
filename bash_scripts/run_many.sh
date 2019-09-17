for i in {0..4};
do sbatch --dependency=singleton  $1; 
done

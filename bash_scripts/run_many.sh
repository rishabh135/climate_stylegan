for i in {0..10};
do sbatch --dependency=singleton  $1; 
done

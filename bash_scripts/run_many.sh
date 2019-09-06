for i in {0..1};
do sbatch --dependency=singleton  $1; 
done

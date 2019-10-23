for i in {0..2};
do sbatch --dependency=singleton  $1; 
done

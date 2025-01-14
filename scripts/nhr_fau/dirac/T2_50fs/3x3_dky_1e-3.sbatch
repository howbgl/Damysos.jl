#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=09:40:00
#SBATCH -e /home/atuin/b228da/b228da10/dirac/T2_50fs/zeta-M-sweep1/3x3_dky_1e-3/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/dirac/T2_50fs/zeta-M-sweep1/3x3_dky_1e-3/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G 3x3_dky_1e-3.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
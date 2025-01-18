#!/bin/bash -l
#SBATCH --gres=gpu:a100:2
#SBATCH --time=19:00:00
#SBATCH -e /home/atuin/b228da/b228da10/dirac/T2_100fs/zeta-M-sweep1/3x3_dky_1e-2/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/dirac/T2_100fs/zeta-M-sweep1/3x3_dky_1e-2/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G 3x3_dky_1e-2.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
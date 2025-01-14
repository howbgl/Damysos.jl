#!/bin/bash -l
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --array=3
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep1/9x9_kymax_first_1e-2/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep1/9x9_kymax_first_1e-2/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G 9x9_kymax_first_1e-2.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
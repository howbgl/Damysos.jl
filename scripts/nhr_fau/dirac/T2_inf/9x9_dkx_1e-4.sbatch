#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:10:00
#SBATCH --array=33-36,38
#SBATCH -e /home/atuin/b228da/b228da10/dirac/T2_inf/9x9_dkx_1e-4/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/dirac/T2_inf/9x9_dkx_1e-4/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto 9x9_dkx_1e-4.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
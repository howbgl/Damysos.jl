#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:10:00
#SBATCH --array=1-81
#SBATCH -e /home/atuin/b228da/b228da10/dirac/T2_inf/converge_r_9x9_dkx/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/dirac/T2_inf/converge_r_9x9_dkx/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto converge_r_9x9_dkx_continue1.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
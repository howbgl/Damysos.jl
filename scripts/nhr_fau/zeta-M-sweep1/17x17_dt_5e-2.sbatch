#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:30:00
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep1/17x17_dt_5e-2/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep1/17x17_dt_5e-2/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto 17x17_dt_5e-2.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
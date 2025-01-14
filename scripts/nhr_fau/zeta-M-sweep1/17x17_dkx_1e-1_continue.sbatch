#!/bin/bash -l
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep1/17x17_dkx_1e-1/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep1/17x17_dkx_1e-1/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto 17x17_dkx_1e-1_continue.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
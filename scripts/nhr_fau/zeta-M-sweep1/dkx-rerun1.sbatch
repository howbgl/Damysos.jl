#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:30:00
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep1/dkx2_1e-4/rerun/slurm-%j.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep1/dkx2_1e-4/rerun/slurm-%j.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G dkx-rerun1.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
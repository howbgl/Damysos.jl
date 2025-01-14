#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:20:00
#SBATCH --array=1-3
#SBATCH -e /home/atuin/b228da/b228da10/quadratic/zeta-M-sweep1/17x17_dt_1e-4/rerun1-slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/quadratic/zeta-M-sweep1/17x17_dt_1e-4/rerun1-slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G 17x17_dt-sweep1-quad_1e-4-rerun.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
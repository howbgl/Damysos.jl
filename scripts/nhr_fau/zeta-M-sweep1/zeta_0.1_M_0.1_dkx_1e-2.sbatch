#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:10:00
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep1/zeta_0.1_M_0.1_dkx_1e-2/slurm-%j.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep1/zeta_0.1_M_0.1_dkx_1e-2/slurm-%j.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G zeta_0.1_M_0.1_dkx_1e-2.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
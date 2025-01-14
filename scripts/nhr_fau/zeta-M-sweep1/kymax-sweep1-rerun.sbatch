#!/bin/bash -l
#SBATCH --gres=gpu:a100:8
#SBATCH --time=23:15:00
#SBATCH --array=1-3
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep1/dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2_kymax_1e-2_atol_1e-6/rerun2-slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep1/dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2_kymax_1e-2_atol_1e-6/rerun2-slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project kymax-sweep1-rerun.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
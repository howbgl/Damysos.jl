#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:10:00
#SBATCH --array=1-2
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep1/17x17_dky_1e-2/slurm-resume-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep1/17x17_dky_1e-2/slurm-resume-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G 17x17_dky-sweep1_1e-2_resume.jl

exit_code=$?

if [ $exit_code -eq 0 ]
then
  echo "Convergence goal achieved."
else
  echo "FAILURE: Convergence goal not achieved."
fi

echo "JOB FINISHED"
exit $exit_code
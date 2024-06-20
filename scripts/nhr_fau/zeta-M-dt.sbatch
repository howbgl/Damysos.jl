#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00
#SBATCH --array=4,9,10,24,26
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G scripts/nhr_fau/zeta-M-sweep1_dt.jl
echo "JOB FINISHED"

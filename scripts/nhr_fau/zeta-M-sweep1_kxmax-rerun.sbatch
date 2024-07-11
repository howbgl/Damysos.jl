#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:40:00
#SBATCH --array=1-32
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep/1d/kxmax2/slurm-%A_%a.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep/1d/kxmax2/slurm-%A_%a.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G zeta-M-sweep1_kxmax-rerun.jl
echo "JOB FINISHED"

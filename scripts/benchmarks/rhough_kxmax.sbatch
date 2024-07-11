#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=18:30:00
#SBATCH -e /home/atuin/b228da/b228da10/rhough_kxmax/slurm-%j.err
#SBATCH -o /home/atuin/b228da/b228da10/rhough_kxmax/slurm-%j.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G rhough_kxmax.jl
echo "JOB FINISHED"

#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:40:00
#SBATCH -e /home/atuin/b228da/b228da10/gpu-memtest/slurm-%j.err
#SBATCH -o /home/atuin/b228da/b228da10/gpu-memtest/slurm-%j.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project --heap-size-hint=60G -t auto gpu-memtest.jl
echo "JOB FINISHED"

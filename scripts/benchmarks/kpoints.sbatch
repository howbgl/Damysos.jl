#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:10:00
#SBATCH -e /home/hpc/b228da/b228da10/dev_damysos/scripts/benchmarks/slurm-%j.err
#SBATCH -o /home/hpc/b228da/b228da10/dev_damysos/scripts/benchmarks/slurm-%j.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project --heap-size-hint=60G -t auto kpoints.jl
echo "JOB FINISHED"

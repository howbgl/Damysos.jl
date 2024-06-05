#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH -e /home/atuin/b228da/b228da10/kchunk_bench/slurm-%j.err
#SBATCH -o /home/atuin/b228da/b228da10/kchunk_bench/slurm-%j.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto scripts/nhr_fau/kchunk_bench.jl
echo "JOB FINISHED"

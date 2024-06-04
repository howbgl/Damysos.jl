#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-test/slurm-%j.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-test/slurm-%j.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto scripts/nhr_fau/zeta-M-sweep1.jl
echo "JOB FINISHED"

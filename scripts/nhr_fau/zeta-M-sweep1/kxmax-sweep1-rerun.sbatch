#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:30:00
#SBATCH -e /home/atuin/b228da/b228da10/zeta-M-sweep1/zeta-M-sweep1/dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2/zeta=0.1_M=10.0/rerun-slurm-%j.err
#SBATCH -o /home/atuin/b228da/b228da10/zeta-M-sweep1/zeta-M-sweep1/dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2/zeta=0.1_M=10.0/rerun-slurm-%j.out

export PATH=$PATH:/home/hpc/b228da/b228da10/.juliaup/bin

julia --project -t auto --heap-size-hint=60G kxmax-sweep1-rerun.jl
echo "JOB FINISHED"

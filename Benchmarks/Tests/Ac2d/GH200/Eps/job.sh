#!/bin/sh
#SBATCH --account=nn4680k     # Your project
#SBATCH --job-name=gpuport    # Help you identify your jobs
##SBATCH --partition=accel
##SBATCH --gpus=1              # Total number of GPUs (incl. all memory of that GPU)
#SBATCH --time=0-2:0:0        # Total maximum runtime. In this case 1 day
#SBATCH --cpus-per-task=128    # All CPU cores of one Grace-Hopper card
#SBATCH --mem-per-cpu=64    # Amount of CPU memory

source $HOME/Etc/modules.sh
./mk.sh cuda

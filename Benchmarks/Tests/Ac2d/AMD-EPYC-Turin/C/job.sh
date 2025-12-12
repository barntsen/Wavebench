#!/bin/sh


#SBATCH --account=nn4680k     # Your project
#SBATCH --job-name=gpuport    # Help you identify your jobs
##SBATCH --partition=accel
##SBATCH --gpus=1              # Total number of GPUs (incl. all memory of that GPU)
#SBATCH --time=1-0:0:0        # Total maximum runtime. In this case 1 day
#SBATCH --cpus-per-task=1    # All CPU cores of one Grace-Hopper card
#SBATCH --mem-per-cpu=3G    # Amount of CPU memory

./mk.sh

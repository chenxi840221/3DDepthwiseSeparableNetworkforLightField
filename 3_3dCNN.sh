#!/bin/bash
#PBS -q gpuvolta
#PBS -l ncpus=48,ngpus=4
#PBS -l walltime=48:00:00,mem=100GB
#PBS -l wd

module load python3/3.7.4

export PYTHONPATH=/home/549/xc0957/.local/lib/python3.7/site-packages/:$PYTHONPATH
python3 myepi.py > $PBS_JOBID.log


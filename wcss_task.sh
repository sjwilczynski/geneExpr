#!/bin/bash
#PBS -q main
#PBS -l walltime=120:00:00
#PBS -l select=1:ncpus=20:mem=20000MB
#PBS -l software=R
#PBS -m be
 
# move to task's directory
cd $PBS_O_WORKDIR

module load r
 
# run the program
# with saving output to a file -- very important (without it the task can be killed) 
Rscript mlcc_run_script.R 8 20 40 30 "c(20, 50, 75, 100, 150, 200, 500)"  >& task_output.txt

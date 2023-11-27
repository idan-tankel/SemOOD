#!/bin/bash

#SBATCH -J LLAMA_%a
#SBATCH --time=08:00:00
#SBATCH --mem=48G
/usr/bin/env /home/idanta/anaconda3/envs/lama/bin/python SEED-Bench/LLama2.py --question_type_id $SLURM_ARRAY_TASK_ID

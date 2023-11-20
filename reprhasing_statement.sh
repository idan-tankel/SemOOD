#!/bin/bash

#SBATCH -J LLAMA_%a
#SBATCH --time=06:00:00
#SBATCH --mem=24G
/usr/bin/env /home/idanta/anaconda3/envs/lama/bin/python SEED-Bench/LLaMA_statements.py --question_type_id $SLURM_ARRAY_TASK_ID

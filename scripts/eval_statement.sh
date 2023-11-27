#!/bin/bash

#SBATCH -J SEED_Bench_statements%a
#SBATCH --time=06:00:00
#SBATCH --mem=48G
/usr/bin/env /home/idanta/anaconda3/envs/lama/bin/python SEED-Bench/Blip2_single_statement_eval.py --question_type_id $SLURM_ARRAY_TASK_ID

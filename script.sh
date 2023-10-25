#!/bin/bash

#SBATCH -J SEED_Bench_%a
#SBATCH --time=05:00:00
#SBATCH --mem=48G
/usr/bin/env /home/idanta/anaconda3/envs/lama/bin/python SEED-Bench/BLIP2_eval.py --question_type_id $SLURM_ARRAY_TASK_ID

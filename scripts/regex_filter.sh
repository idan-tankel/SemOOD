#!/bin/bash

#SBATCH -J SEED_Bench_%a
/usr/bin/env /home/idanta/anaconda3/envs/lama/bin/python SEED-Bench/filter_pattern.py --question_type_id $SLURM_ARRAY_TASK_ID

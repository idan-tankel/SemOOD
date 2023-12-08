#!/bin/bash

#SBATCH -J SEED_Bench_rephrasing%a
#SBATCH --output=slurm_logs/%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
/usr/bin/env /home/idanta/anaconda3/envs/lama/bin/python SEED-Bench/InstructBlip_eval_rephrasing.py --question_type_id $SLURM_ARRAY_TASK_ID

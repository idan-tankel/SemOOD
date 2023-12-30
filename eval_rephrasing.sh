#!/bin/bash

#SBATCH -J Eval_reprhasing_%a
#SBATCH --time=06:00:00
#SBATCH --mem=48G
/usr/bin/env /home/idanta/anaconda3/envs/lama/bin/python SEED-Bench/BLIP2_eval_reprhasing.py --question_type_id $SLURM_ARRAY_TASK_ID --dataset_path /home/idanta/data/SEED/SEED-Bench-image/reconstruction/rephrased/fully_processed

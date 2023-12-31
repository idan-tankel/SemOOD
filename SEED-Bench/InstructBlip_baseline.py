from evaluator_strategies.InstructBlipModels import InstructBlipBaseline
import argparse
import os
import pandas as pd
from datasets import load_dataset
import wandb
from torch.utils.data import DataLoader
task_names = {"Scene Understanding": 1,
              "Instance Identity": 2,
              "Instance Attributes": 3,
              "Instance Location": 4,
              "Instances Counting": 5,
              "Spatial Relation": 6,
              "Instance Interaction": 7,
              "Visual Reasoning": 8,
              "Text Understanding": 9,
              "Action Recognition": 10,
              "Action Prediction": 11,
              "Procedure Understanding": 12
              }
task_ids = {v: k for k, v in task_names.items()}


def main():
    evaluator = InstructBlipBaseline(root_dir="./data", device="cuda", names=["choice_a", "choice_b", "choice_c", "choice_d"])
    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench/Image_questions.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--question_type_id', default=7, type=int)
    args = parser.parse_args()
    task_name = task_ids.get(args.question_type_id)
    wandb.init(
        # set the wandb project where this run will be logged
        project="SemOOD",
        # track hyperparameters and run metadata
        # dynamic configuration
        # no hyperparameters were tuned
        config={
            "architecture": evaluator.__class__.__name__,
            "question_format": "InstructBlip - Question:.... Answer:...",
            "dataset": f"Seed-Bench_{task_name}",
            "task_index": args.question_type_id,
            "epochs": 1,  # inference no training
        }
    )
    huggingface_data_dir = rf"/net/mraid11/export/data/idanta/SEED/SEED-Bench-image"
    huggingface_dataset = load_dataset("AILab-CVC/SEED-Bench", cache_dir=huggingface_data_dir, data_dir=huggingface_data_dir, split=None)
    huggingface_dataset.with_format("torch")
    dataset = huggingface_dataset["test"]
#    dataset = Dataset.from_json(args.anno_path, field='questions')
    # dataset.with_format("torch")
    # filter the dataset and split by the task type
    if args.question_type_id is not None:
        dataset = dataset.filter(lambda x: int(x['question_type_id']) == args.question_type_id)
    if 'segment' in dataset.features:
        dataset = dataset.remove_columns("segment")
    total = len(dataset)

    baseline_df = pd.read_csv("SEED-Bench/docs/leaderboard.csv")
    result_dict = baseline_df.to_dict(orient='records')[0]
    # log the table as a SCALAR
    wandb.log({"Baseline": result_dict.get(task_name)})
    # loading data
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    scores, acc_percent = evaluator.get_retrieval_scores(joint_loader=data_loader, total_examples_for_task=total)
    wandb.log({"scores(std)": scores.std()})
    wandb.log({"evaluator(acc)": acc_percent})
    wandb.log({"total examples": len(data_loader)})
    wandb.log({"total valid examples": len(data_loader) - evaluator.failed_count})

    # The interface for testing MLLMs
# .*\\n1\W+(.*)\W+\\n2\W+(.*)\W+\\n.*3\W+(.*)\W+\\n4\W+(.*)\W+\\n
# regex for splitting up the captions from the new prompts

if __name__ == '__main__':
    main()
    wandb.finish()

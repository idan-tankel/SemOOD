from evaluator.BLIP2Models import BLIP2HFModelWrapper
import argparse
import os
from datasets import Dataset
import wandb
from torch.utils.data import DataLoader

wandb.init(
    # set the wandb project where this run will be logged
    project="SemOOD",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "BLIP2HFModelWrapper",
        "dataset": "Seed-Bench",
        "epochs": 1,
    }
)


def main():
    evaluator = BLIP2HFModelWrapper(root_dir="./data", device="cuda")
    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench/Statements.json')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    dataset = Dataset.from_json(args.anno_path, field='questions')
    dataset.with_format("torch")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    scores = evaluator.get_retrieval_scores(joint_loader=data_loader)
    wandb.log({"scores(std)": scores.std()})

    # The interface for testing MLLMs


if __name__ == '__main__':
    main()
    wandb.finish()

from evaluator.BLIP2Models import BLIP2HFModelWrapper
import argparse
import os
import json
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader


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
    evaluator.get_retrieval_scores_batched(joint_loader=data_loader)

    # The interface for testing MLLMs


if __name__ == '__main__':
    main()

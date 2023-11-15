from datasets import load_from_disk
import re
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--question_type_id', default=1, type=int)
args = parser.parse_args()


def update_dataset(example):
    pattern = r'.*^1\W+(.*)^2\W+(.*)^3\W+(.*)^4\W+([^\n]*)'
    formatted_output = example["new"]
    match = re.search(pattern, formatted_output, re.MULTILINE | re.DOTALL)
    if match:
        example['new_1'] = match.group(1)
        example['new_2'] = match.group(2)
        example['new_3'] = match.group(3)
        example['new_4'] = match.group(4)
    else:
        example["new_1"] = example["new_2"] = example["new_3"] = example["new_4"] = None
        print("NotMatchedError")
    return example


def main():
    save_dir = rf'/net/mraid11/export/data/idanta/SEED/SEED-Bench-image/rephrased/{args.question_type_id}'
    dataset = load_from_disk(save_dir)
    dataset = dataset.map(update_dataset)
    dataset.save_to_disk("/net/mraid11/export/data/idanta/SEED/SEED-Bench-image/fully_processed")


if __name__ == "__main__":
    main()

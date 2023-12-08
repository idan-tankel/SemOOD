from math import e
import os
from datasets import load_from_disk
import re
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--question_type_id', default=7, type=int)
parser.add_argument('--huggingface_data_folder',default='/net/mraid11/export/data/idanta/SEED/SEED-Bench-image/4_choice_at_once/')
args = parser.parse_args()


def update_dataset(example):
    pattern = r'.*^1\W+(.*)^2\W+(.*)^3\W+(.*)^4\W+([^\n]*)'
    another_pattern = r'.*\(?A\)?\W+(.*)^\(?B\)?\W+(.*)^\(?C\)?\W+(.*)^\(?D\)?\W+([^\n]*)'
    new_pattern = r'.*^•\W?(.*)\n^•\W?(.*)\n^•\W?(.*)\n^•\W?(.*)'
    choice_pattern = r'.*^•\W?(.*)^•\W?(.*)^•\W?(.*)^•\W?(.*)'
    star_pattern = r'.*^\*(.*)\n^\*(.*)\n^\*(.*)\n^\*(.*)'
    one_pattern = r'.*(?:^•?\W?Choice[_ ][A-Da-d]\W?(.*)\n)(?:^•?\W?Choice[_ ][A-Da-d]\W?(.*)\n)(?:^•?\W?Choice[_ ][A-Da-d]\W?(.*)\n)(?:^•?\W?Choice[_ ][A-Da-d]\W?(.*))'
    formatted_output = example["new"]
    match = re.search(pattern, formatted_output, re.MULTILINE | re.DOTALL)
    second_match = re.search(another_pattern, formatted_output, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    third_match = re.search(new_pattern, formatted_output, re.MULTILINE | re.DOTALL)
    forth_match = re.search(choice_pattern, formatted_output, re.MULTILINE | re.DOTALL)
    fith = re.search(star_pattern, formatted_output, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    six = re.search(one_pattern, formatted_output, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if match:
        example['new_1'] = match.group(1)
        example['new_2'] = match.group(2)
        example['new_3'] = match.group(3)
        example['new_4'] = match.group(4)
        # fully parsed, return example and continue
        return example
    elif second_match:
        example['new_1'] = second_match.group(1)
        example['new_2'] = second_match.group(2)
        example['new_3'] = second_match.group(3)
        example['new_4'] = second_match.group(4)
        return example
    elif third_match:
        example['new_1'] = third_match.group(1)
        example['new_2'] = third_match.group(2)
        example['new_3'] = third_match.group(3)
        example['new_4'] = third_match.group(4)
        return example
    elif forth_match:
        example['new_1'] = forth_match.group(1)
        example['new_2'] = forth_match.group(2)
        example['new_3'] = forth_match.group(3)
        example['new_4'] = forth_match.group(4)
        return example
    elif fith:
        example['new_1'] = fith.group(1)
        example['new_2'] = fith.group(2)
        example['new_3'] = fith.group(3)
        example['new_4'] = fith.group(4)
        return example
    elif six:
        example['new_1'] = six.group(1)
        example['new_2'] = six.group(2)
        example['new_3'] = six.group(3)
        example['new_4'] = six.group(4)
        return example
    else:
        example["new_1"] = example["new_2"] = example["new_3"] = example["new_4"] = None
    return example


def main():
    save_dir = rf'{args.huggingface_data_folder}/rephrased/{args.question_type_id}'
    dataset = load_from_disk(save_dir)
    dataset = dataset.map(update_dataset)
    fully_processed_save_dir = f'{args.huggingface_data_folder}/fully_processed/{args.question_type_id}'
    os.makedirs(fully_processed_save_dir, exist_ok=True)
    dataset.save_to_disk(fully_processed_save_dir)


if __name__ == "__main__":
    main()

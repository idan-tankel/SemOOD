from evaluator_strategies.BLIP2Models import Blip2AnswerByQuestionRephrasing
from argparse import ArgumentParser
import os
from datasets import load_dataset, load_from_disk
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
    parser = ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench/Image_questions.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--question_type_id', default=1, type=int)
    args = parser.parse_args()
    task_name = task_ids.get(args.question_type_id)
    evaluator = Blip2AnswerByQuestionRephrasing(root_dir="./data", device="cuda", names=["statement_1", "statement_2", "statement_3", "statement_4"], task_name=task_name)
    wandb.init(
        # set the wandb project where this run will be logged
        project="SemOOD",
        # track hyperparameters and run metadata
        # dynamic configuration
        # no hyperparameters were tuned
        config={
            "architecture": evaluator.__class__.__name__,
            "question_format": "rephrasing using 4 captions",
            "dataset": f"Seed-Bench_{task_name}",
            "task_index": args.question_type_id,
            "epochs": 1,  # inference no training
        }
    )
    huggingface_data_dir = rf"/net/mraid11/export/data/idanta/SEED/SEED-Bench-image"
    huggingface_dataset = load_dataset("AILab-CVC/SEED-Bench", cache_dir=huggingface_data_dir, data_dir=huggingface_data_dir, split=None)
    dataset = huggingface_dataset["test"]
    dataset = load_from_disk(rf"/net/mraid11/export/data/idanta/SEED/SEED-Bench-image/rephrased_single_statement/{args.question_type_id}")
    huggingface_dataset.with_format("torch")
    # dataset = Dataset.from_json(args.anno_path, field='questions')
    # dataset.with_format("torch")
    # filter the dataset and split by the task type
    if args.question_type_id is not None:
        dataset = dataset.filter(lambda x: int(x['question_type_id']) == args.question_type_id)
    if 'segment' in dataset.features:
        dataset = dataset.remove_columns("segment")

    total_examples_for_task = len(dataset)
    if "new_1" in dataset.features:
        # filter to only the new examples
        dataset = dataset.filter(lambda x: x["statement_1"] is not None)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # loading data
    assert len(data_loader) > 0, "data_loader_is_empty. check the filters"
    # since the dataset examples without new_1 (did not parse well) are not kept for the statistics
    evaluator.failed_count += (total_examples_for_task - len(data_loader))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    scores, acc_percent = evaluator.get_retrieval_scores(joint_loader=data_loader, total_examples_for_task=total_examples_for_task)
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

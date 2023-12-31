from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--question_type_id', default=7, type=int)
parser.add_argument('--huggingface_data_dir', default=r"/home/idanta/data/SEED/SEED-Bench-image")
parser.add_argument('--save_dir', default=r"/home/idanta/data/SEED/SEED-Bench-image/new_4_choice_at_once")
args = parser.parse_args()

model = "meta-llama/Llama-2-7b-chat-hf"


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
base_prompt = """<<SYS>>You are converting a multi-choice question about the image into a 4-choice possible sentences. return new list of choices, based on the previous ones, matching the initial question. keep them on the same original order. Do NOT give an extra information that is not written.<</SYS>>[INST]User:{text}[/INST]\nAssistant: """


def generate(text: str):
    input = base_prompt.format(text=text)
    # '"question": "What type of building is in the image?"\n"answer": "A hotel"\n“Statement”: "The type of the building in the image is a hotel."\n"question": "What type of building is in the image?"\n"answer": "A house"\n“Statement”: "The type of the building in the image is a house."\n"question": "How many towels are in the image?"\n"answer": "One"\n“Statement”: “There is one towel in the image.”\n"question": "How many towels are in the image?"\n"answer": "Two"\n“Statement”: “There are Two towels in the image.”\n"question": "How many towels are in the image?"\n"answer": "Three"\n“Statement”: “There are Three towels in the image.”\n"question": "Based on the scene, what can be inferred about the current state of the game?"\n"answer": "The game is ongoing, and the player is making an impressive play."\n"Statement": "',
    sequences = pipeline(
        input,
        do_sample=True,
        top_k=50,
        num_return_sequences=1,
        max_new_tokens=200,
        return_full_text=False,
        temperature=0.9,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )
    for seq in sequences:
        print(f"LLama's 🦙 answer: {seq['generated_text']}")
    return sequences[0]["generated_text"]


def generate_for_example(example):
    prompt = f'''"question": "{example['question']}",\n\
            "choice_a": "{example['choice_a']}",\n\
            "choice_b": "{example['choice_b']}",\n\
            "choice_c": "{example['choice_c']}",\n\
            "choice_d": "{example['choice_d']}"'''
    example["new"] = generate(prompt)
    return example


# loop over the generate for the whole dataset
huggingface_data_dir = args.huggingface_data_dir
save_dir = os.path.join(args.save_dir, str(args.question_type_id))
os.makedirs(save_dir, exist_ok=True)
huggingface_dataset = load_dataset("AILab-CVC/SEED-Bench", data_dir=huggingface_data_dir, split=None)
huggingface_dataset.with_format("torch")
dataset = huggingface_dataset["test"]
dataset = dataset.filter(lambda x: int(x['question_type_id']) == args.question_type_id)
# see example here
# https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.filter
# dataset = dataset.select(range(100))
# dataset = Dataset.from_json(args.anno_path, field='questions')
# dataset.with_format("torch")
# filter the dataset and split by the task type

# loading data
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
# looping over the dataset, applying generate function using map
new_dataset = dataset.map(generate_for_example)
new_dataset.info.description = rf"was generated using prompt {base_prompt}"
new_dataset.save_to_disk(save_dir)
improved_dataset = load_from_disk(save_dir)

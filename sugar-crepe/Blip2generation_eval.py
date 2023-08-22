import torch
from PIL import Image
import numpy
import os
from transformers import Blip2Processor,Blip2ForConditionalGeneration
from tqdm import tqdm
import json
import wandb
import pandas as pd
# import pytorch_lightning as pl


def load_model(model_location="Salesforce/blip2-opt-2.7b",device='cuda'):
    preprocessor = Blip2Processor.from_pretrained(model_location)
    model = Blip2ForConditionalGeneration.from_pretrained(model_location)
    model.to(device)
    return model, preprocessor



def local_dataset(data_root) -> dict:
    """load local dataset from json annotation files

    Args:
        data_root (`str`): loacl dataset root path

    Returns:
        `dict`: dataset dict
    """
    data_dict = {
        'add_obj'    : f'{data_root}/add_obj.json',
        'add_att'    : f'{data_root}/add_att.json',
        'replace_obj': f'{data_root}/replace_obj.json',
        'replace_att': f'{data_root}/replace_att.json',
        'replace_rel': f'{data_root}/replace_rel.json',
        'swap_obj'   : f'{data_root}/swap_obj.json',
        'swap_att'   : f'{data_root}/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        assert os.path.exists(data_path), rf"{data_path} does not exist"
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))
    return dataset

def inference_loop(dataset,image_root, model:Blip2ForConditionalGeneration, preprocessor: Blip2Processor, device):
    """
    inference_loop run all the images throgh the model and collect scores Tensor
    one epoch of the dataset

    Args:
        dataset (_type_): _description_
        model (_type_): _description_
        preprocessor (_type_): _description_
        device (_type_): _description_
    """    
    model.eval()
    metrics = {}
    for dataset_type, sub_dataset in dataset.items():
        count_true = 0
        for i, data in tqdm(sub_dataset.items(), desc=f'evaluating {dataset_type}'):
            with torch.no_grad():
                #
                print(f"Case: {dataset_type}, {i}")
                # print(f"Input: {string_index['input']}")

                # print(f"Target: {string_index['target']}")
                # input_ids is to tokenize the text
                image = Image.open(os.path.join(image_root, data['filename']))
                # display(image)
                pos_text,neg_text = data['caption'],data['negative_caption']
                random = numpy.random.choice([0,1])
                if random == 0:
                    prompt = f"Question: The following sentence is correct: (A).{pos_text} (B).{neg_text}. Answer:"
                else:
                    prompt = f"Question: The following sentence is correct: (A).{neg_text} (B).{pos_text}. Answer:"
                wright_answer = "A" if random == 0 else "B"                                    
                inputs = preprocessor(images=image, text=prompt,return_tensors='pt').to(device)
                # the last 2 kwargs are configuration for the GenerationMixin
                # the scores are the ones for the next token prediction (argmax)
                # this is since the greedy algorithm is used
                # output_scores=True, return_dict_in_generate=True
                output_ids = model.generate(**inputs)
                output = preprocessor.decode(output_ids[0,1:])
                print(f"Output: {output}")
                print("=====================================")
                true = output[0:1] == f' {wright_answer}' or output[0:4] == f' ({wright_answer})'
                if true:
                    count_true += 1
                print(true)
        print(count_true / float(i))
        acc = 100.0*count_true / float(i)
        metrics.update({dataset_type: acc})
    print(metrics)
    # metrics = {k: v.to(device='cpu', non_blocking=True).item().numpy() for k, v in metrics.items ()}
    metrics = pd.DataFrame(metrics)
    print(f"Dump results to: {os.path.join(args.output, f'{args.model}-{args.pretrained}.json')}")
    metrics.to_json(os.path.join(args.output, f'{args.model}-{args.pretrained}.json'),indent=4)



def main():
    image_root = "sugar-crepe/data/coco/images/val2017"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load dataset
    dataset = local_dataset("sugar-crepe/data")
    model, preprocessor = load_model()
    inference_loop(dataset,image_root,model,preprocessor,device)



if __name__ == '__main__':
    main()


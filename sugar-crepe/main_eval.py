import argparse
import json
import os
import pandas as pd
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
import torch
from PIL import Image
from tqdm import tqdm

models = [
    ('RN50', 'openai'),
    ('RN101', 'openai'),
    ('RN50x4', 'openai'),
    ('ViT-B-32', 'openai'),
    ('RN50x16', 'openai'),
    ('RN50x64', 'openai'),
    ('ViT-L-14', 'openai'),
    ('blip2-opt-2.7b', 'Salesforce'),
    # ('ViT-B-32-quickgelu', 'datacomp_s_s13m_b4k'),
    # ('ViT-B-32-quickgelu', 'datacomp_m_s128m_b4k'),
    # ('ViT-B-16', 'datacomp_l_s1b_b8k'),
    # ('ViT-L-14', 'datacomp_xl_s13b_b90k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-g-14', 'laion2b_s12b_b42k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k'),
    ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
    ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
    ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),
]


def load_model(args, pretrained, device):
    # model, _, transform = open_clip.create_model_and_transforms(
    #     model_name=args.model,
    #     pretrained=pretrained,
    #     cache_dir=args.model_cache_dir,
    #     device=device
    # )
    # load other models not from openCLIP
#   from PIL import Image

#device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, tokenizer = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
    transform = lambda x: x  # identity
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer, transform


@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model,preprocessor, tokenizer, transform, device):
    try:
        preprocessed_image = preprocessor["eval"](image).to(device)
        if preprocessed_image.dim() == 3:
            preprocessed_image.unsqueeze_(0)
    except RuntimeError:
        return False

    pos_text_tokenized = tokenizer["eval"](pos_text)
    neg_text_tokenized = tokenizer["eval"](neg_text)
    pos_itc_output = model({"image":preprocessed_image,"text_input":pos_text_tokenized},match_head="itc")
    neg_itc_output = model({"image":preprocessed_image,"text_input":neg_text_tokenized},match_head="itc")

    return pos_itc_output > neg_itc_output

    # pos_text_tokenized = tokenizer(images=None, text=pos_text, return_tensors="pt")
    # pos_text_tokenized.to(device).to(torch.float16)
    # # crucial, see https://github.com/huggingface/blog/pull/900/commits/2e69855bee6217a9fc1a4666bea16d2b54c31c7a
    # # and https://github.com/huggingface/blog/pull/900
    # # pos_text_embedding = model(**pos_text, normalize=True)
    # neg_text_tokenized = tokenizer(images=None, text=neg_text, return_tensors="pt").to(device)
    # neg_text_tokenized.to(device).to(torch.float16)
    # # neg_text_embedding = model.encode_text(neg_text, normalize=True)
    # # image_embedding = model.encode_image(transform(image).unsqueeze(dim=0).to(device), normalize=True)
    # # since that is not clip, do only BLIP evaluation
    # pos_text_outputs = model.get_text_features(**neg_text_tokenized, output_hidden_states=True,output_attentions=True)
    # neg_text_outputs = model.get_text_features(**pos_text_tokenized, output_hidden_states=True,output_attentions=True)
    # pos_text_features = pos_text_outputs['hidden_states'][-1][:, 0, :]
    # neg_text_features = neg_text_outputs['hidden_states'][-1][:, 0, :]
    # # l1 distance to the image features
    # image_tensors = tokenizer(images=image, return_tensors="pt")
    # image_tensors.to(device,torch.float16)
    # attention_mask = neg_text_outputs
    # # see https://github.com/salesforce/LAVIS/blob/f982acc73288408bceda2d35471a8fcf55aa04ca/lavis/models/blip2_models/blip2_image_text_matching.py#L76
    # image_features = model.get_qformer_features(**image_tensors)(**neg_text_tokenized)
    # itm_embeddings = image_features.last_hidden_state
    # itm_logits = self.tm

    # find the close
    pos_text_features /= pos_text_features.norm(dim=-1, keepdim=True)
    neg_text_features /= neg_text_features.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)


def evaluate(image_root, dataset, model,preprocessor, tokenizer, transform, device):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            image_path = os.path.join(image_root, data['filename'])
            image = Image.open(image_path)
            correct = text_retrieval(data['caption'], data['negative_caption'], image, model,preprocessor, tokenizer, transform, device)
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="RN50", help="Model architecture to use from OpenCLIP")
    parser.add_argument('--pretrained', type=str, default="openai", help="Model checkpoint name to use from OpenCLIP")
    parser.add_argument('--model_cache_dir', default=None, type=str, help="Directory to where downloaded models are cached")
    parser.add_argument('--output', type=str, default=None, help="Directory to where results are saved")

    parser.add_argument('--coco_image_root', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--all', action="store_true", default=False, help="Whether to test all the pretrained models in the paper")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dict = {
        'add_obj'    : f'{args.data_root}/add_obj.json',
        'add_att'    : f'{args.data_root}/add_att.json',
        'replace_obj': f'{args.data_root}/replace_obj.json',
        'replace_att': f'{args.data_root}/replace_att.json',
        'replace_rel': f'{args.data_root}/replace_rel.json',
        'swap_obj'   : f'{args.data_root}/swap_obj.json',
        'swap_att'   : f'{args.data_root}/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        assert os.path.exists(data_path), rf"{data_path} does not exist"
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))

    os.makedirs(args.output, exist_ok=True)
    print(f"Evaluating {args.model}-{args.pretrained}")

    model, preprocessor, tokenizer, transform = load_model(args, args.pretrained, device)

    metrics = evaluate(args.coco_image_root, dataset, model,preprocessor, tokenizer, transform, device)
    print(metrics)
    metrics = {k: v.to(device='cpu', non_blocking=True).item().numpy() for k, v in metrics.items ()}
    metrics = pd.DataFrame(metrics)
    print(f"Dump results to: {os.path.join(args.output, f'{args.model}-{args.pretrained}.json')}")
    metrics.to_json(os.path.join(args.output, f'{args.model}-{args.pretrained}.json'),indent=4)

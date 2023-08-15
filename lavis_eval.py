import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

model, preprocessor, tokenizer = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)


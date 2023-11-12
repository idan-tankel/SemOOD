# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-generation", model="liuhaotian/llava-v1.5-7b")
# trigger download of the model into cache
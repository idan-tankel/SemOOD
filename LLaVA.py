
# Solved with following python script

from transformers import AutoConfig, LlamaConfig 
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
class LlavaConfig(LlamaConfig):
    model_type = "llava"
AutoConfig.register("llava", LlavaConfig)
model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-llama-2-13b-chat-lightning-preview")
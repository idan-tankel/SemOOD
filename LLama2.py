from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"


from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    '"question": "What type of building is in the image?"\n"answer": "A hotel"\n“Statement”: "The type of the building in the image is a hotel."\n"question": "What type of building is in the image?"\n"answer": "A house"\n“Statement”: "The type of the building in the image is a house."\n"question": "How many towels are in the image?"\n"answer": "One"\n“Statement”: “There is one towel in the image.”\n"question": "How many towels are in the image?"\n"answer": "Two"\n“Statement”: “There are Two towels in the image.”\n"question": "How many towels are in the image?"\n"answer": "Three"\n“Statement”: “There are Three towels in the image.”\n"question": "Based on the scene, what can be inferred about the current state of the game?"\n"answer": "The game is ongoing, and the player is making an impressive play."\n"Statement": "',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"LLama's answer: {seq['generated_text']}")

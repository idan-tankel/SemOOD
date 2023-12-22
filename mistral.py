from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")


example = {
    "answer": "B",
    "choice_a": "Person",
    "choice_b": "Yellow and blue sculpture",
    "choice_c": "Ball",
    "choice_d": "All of the above",
    "data_id": "415046_4041589337",
    "data_type": "image",
    "question": "Which of the following objects are detected in the image?",
    "question_id": "46773",
    "question_type_id": 2
}


message = rf'''"question": "{example['question']}",\n\
            "choice_a": "{example['choice_a']}",\n\
            "choice_b": "{example['choice_b']}",\n\
            "choice_c": "{example['choice_c']}",\n\
            "choice_d": "{example['choice_d']}"'''

instruction = "You are converting a multi-choice question about the image into a 4-choice possible sentences. return new list of choices based on the previous ones, matching the initial question. keep them on the same original order. Do NOT give an extra information that is not written"
messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": f"{message}"}
]



prompt = f"<s>[INST] {instruction} [/INST] Hello! how can I help you</s>[INST] {message} [/INST]"

# according to the content of huggingface.co standartization
# see chat templates here https://huggingface.co/docs/transformers/main/chat_templating
encodeds = tokenizer.encode(prompt, return_tensors="pt")
# encodeds = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

model_inputs = encodeds.to(device)  # type: ignore
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

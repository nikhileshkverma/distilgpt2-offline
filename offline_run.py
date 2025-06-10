from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model & tokenizer from local folder
model_name = "./distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate from prompt
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=50)

# Print output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


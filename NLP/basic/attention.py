from torch import nn
from transformers import AutoConfig, AutoTokenizer


model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = 'time files like an arrow'
inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
print(inputs.input_ids)

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name = "dpo/final_checkpoint"

print("Loading model...")
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Generating...")
inputs = tokenizer.encode("Question: Please write a function in Python that performs bubble sort.\n\nAnswer:", return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
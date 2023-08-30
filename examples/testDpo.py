import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name = "dpo/final_checkpoint"

model = AutoPeftModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer.encode("Question: Please write a function in Python that performs bubble sort.\n\nAnswer:", return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
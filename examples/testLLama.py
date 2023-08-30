from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name_or_path = "/home/ec2-user/SageMaker/llama-tune/trl/.cache/kashif/models--kashif--stack-llama-2/snapshots/28a206689c0097738177840a40e455a308db2d7d"  #path/to/your/model/or/name/on/hub
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    cache_dir=".cache/kashif/",
    device_map={"": 0},
)
tokenizer = AutoTokenizer.from_pretrained(
    "kashif/stack-llama-2",
    trust_remote_code=True,
    cache_dir=".cache/llama/tokenizer",)
print("Model loaded successfully!")
inputs = tokenizer.encode("This movie was really", return_tensors="pt").to("cuda:0")
print("Generating...")
outputs = model.generate(inputs)
print("Generated text:")
print(tokenizer.decode(outputs[0]))
#循环等待用户输入
while True:
    print("请输入：")
    text = input()
    if text == "quit":
        break
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(inputs)
    print("Generated text:")
    print(tokenizer.decode(outputs[0]))


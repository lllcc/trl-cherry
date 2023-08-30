from datasets import load_dataset

streaming = True
num_workers = 4
print("loading dataset")
dataset = load_dataset(
    "lvwerra/stack-exchange-paired",
    data_dir="data/finetune",
    split="train",
    cache_dir="./.cache/huggingface/datasets/sft",
    use_auth_token=True,
    num_proc=num_workers if not streaming else None,
    streaming=streaming,
)
print("dataset load success")
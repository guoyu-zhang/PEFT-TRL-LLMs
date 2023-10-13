from datasets import load_dataset

dataset = load_dataset("Dahoas/full-hh-rlhf")

print(dataset['train'][0])

from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
import wandb

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, Trainer, TrainingArguments, AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed, RewardTrainer
from trl.core import LengthSampler

wandb.init()


config = PPOConfig(
    # model_name="meta-llama/Llama-2-7b",
    model_name="bert-base-uncased",
    learning_rate=1.41e-5,
    log_with="wandb",
)

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size":16}

##defininig the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(config.model_name)

dataset = load_dataset("Dahoas/full-hh-rlhf")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def format_func(row):
  kwargs = {"padding": "max_length",
              "truncation": True,
              "max_length": 256,
              "return_tensors": "pt"
              }
  
  prompt_and_chosen = row['prompt'] + "\n" + row['chosen']
  prompt_and_rejected = row['prompt'] + "\n" + row['rejected']
  
  tokens_chosen = tokenizer.encode_plus(prompt_and_chosen, **kwargs)
  tokens_rejected = tokenizer.encode_plus(prompt_and_rejected, **kwargs)

  return {
      "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
      "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
  }
  
formatted_dataset = dataset.map(format_func)

### Loading the TRL reward trainer and training the trainer
training_args = TrainingArguments(
        output_dir="rm_checkpoint/",
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        learning_rate=1e-5,
        save_total_limit=1,
        no_cuda=True
    )

trainer = RewardTrainer(model=model,
                        tokenizer=tokenizer,
                        train_dataset=formatted_dataset['train'],
                        eval_dataset=formatted_dataset['test'],
                        args=training_args
                        )
trainer.train()

trainer.save_model("rm_model/")
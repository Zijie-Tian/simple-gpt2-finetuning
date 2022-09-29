import os
import collections
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, GPTNeoXConfig

LOCAL_RANK  =   int(os.environ.get('RANK'))
if not LOCAL_RANK  ==  0:
    import sys
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

torch.manual_seed(42)
tokenizer = GPTNeoXTokenizerFast.from_pretrained("pretrained_model", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
config = GPTNeoXConfig.from_pretrained("pretrained_model")
model = GPTNeoXForCausalLM.from_pretrained("pretrained_model", config=config)
training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, warmup_steps=10,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True, deepspeed='./zero3_config.json')


class TextDataset(torch.utils.data.Dataset):
    labels = {
        "business": 0,
        "entertainment": 1,
        "sport": 2,
        "tech": 3,
        "politics": 4
    }

    def __init__(self, df, tokenizer, rank=0, world_size=1):
        self.labels = [TextDataset.labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in df['text']]
        
        # Split the dataset for different ranks
        # self.labels = self.labels[len(self.labels) // world_size * rank:(len(self.labels) // world_size * (rank + 1) if rank < world_size - 1 else len(self.labels))]
        # self.texts = self.texts[len(self.texts) // world_size * rank:(len(self.texts) // world_size * (rank + 1) if rank < world_size - 1 else len(self.texts))]
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

train_dataset = TextDataset(pd.read_csv('./data/train/train.csv'), tokenizer)
val_dataset = TextDataset(pd.read_csv('./data/val/val.csv'), tokenizer)
test_dataset = TextDataset(pd.read_csv('./data/test/test.csv'), tokenizer)

Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])}).train()
generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                bos_token='<|startoftext|>',
                                eos_token='<|endoftext|>', pad_token='<|pad|>',
                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                                                            

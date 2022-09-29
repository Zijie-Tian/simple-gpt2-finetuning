import os
import collections
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import AutoTokenizer, DataCollator, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, GPTNeoXConfig, GPT2Config
from transformers import GPTJForCausalLM, GPTJConfig

LOCAL_RANK  =   int(os.environ.get('RANK'))
if not LOCAL_RANK  ==  0:
    import sys
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

class TextDataset(torch.utils.data.Dataset):
    labels = {
        "business": 0,
        "entertainment": 1,
        "sport": 2,
        "tech": 3,
        "politics": 4
    }

    def __init__(self, df, tokenizer, max_length, rank=0, world_size=1):
        self.labels = [TextDataset.labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=max_length,
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

class NetflixDataset(torch.utils.data.Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]



def collator_fn(data):
    # data is a list of (input, label) tuples
    # print(data)
    # print([torch.cat([x[0]['input_ids'] for x in data], 0)])
    ret = {
        "input_ids": torch.cat([x[0]['input_ids'] for x in data], dim=0),
        "attention_mask": torch.cat([x[0]['attention_mask'] for x in data], dim=0),
        "labels": torch.cat([torch.tensor(x[1]).view(1) for x in data], dim=0)
    }
    # print(ret)
    return ret

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
config = GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", config=config)
model.resize_token_embeddings(len(tokenizer))
descriptions = pd.read_csv('./data/train/train.csv')['text']
max_length = max([len(tokenizer.encode(description)) for description in descriptions])
training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=4, per_device_eval_batch_size=4, warmup_steps=10,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True, deepspeed='./zero3_config.json')

train_dataset = TextDataset(pd.read_csv('./data/train/train.csv'), tokenizer, max_length)
val_dataset = TextDataset(pd.read_csv('./data/val/val.csv'), tokenizer, max_length)
test_dataset = TextDataset(pd.read_csv('./data/test/test.csv'), tokenizer, max_length)

# dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)

# for item in dataloader:
#     print(item)
#     break

Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=collator_fn).train()
generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                bos_token='<|startoftext|>',
                                eos_token='<|endoftext|>', pad_token='<|pad|>',
                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                                                            

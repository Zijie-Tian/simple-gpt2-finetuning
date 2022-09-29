import os
from pickletools import optimize
import sys
import re
import errno
import collections
import time
import argparse

import numpy as np
import pandas as pd

import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split

from GPT2NEOX.modeling_gpt_neox import GPTNeoXModel
from GPT2NEOX.configuration_gpt_neox import GPTNeoXConfig
from GPT2NEOX.utils import load_weight, load_weight_recursive
from torch.optim import Adam
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam # teh optimizer of ds 
from deepspeed.runtime.pipe import ProcessTopology
from deepspeed.runtime.pipe.topology import PipelineParallelGrid

import logging
# Initialize the logger and set the level to info
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

rank = int(os.environ['RANK'])
if not rank == 0:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

# Dataset class
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

def get_argparser():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size for training (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--world_size',
                        type=int,
                        default=-1,
                        help='world size passed from distributed launcher')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument(
        "--lr", type=float, default=1e-5, metavar="LR", help="learning rate (default: 1e-5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--hidden-size", type=int, default=768, metavar="HS", help="hidden size (default: 768)")
    parser.add_argument("--max-seq-len", type=int, default=128, metavar="MSL", help="max sequence length (default: 128)")
    

    parser.add_argument("--train-dir", type=str)
    parser.add_argument("--val-dir", type=str)
    parser.add_argument("--test-dir", type=str)

    ## Add deepspeed args
    parser = deepspeed.add_config_arguments(parser)

    return parser

class TextDataset(torch.utils.data.Dataset):
    labels = {
        "business": 0,
        "entertainment": 1,
        "sport": 2,
        "tech": 3,
        "politics": 4
    }

    def __init__(self, df, rank=0, world_size=1):
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


# train data loader
def _get_train_data_loader(batch_size, train_dir, local_rank=0, world_size=1, **kwargs):
    train_df = pd.read_csv(os.path.join(train_dir, "train.csv"))
    train_dataset = TextDataset(train_df, rank=local_rank, world_size=world_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        **kwargs
    )
    return train_dataloader

# val data loader
def _get_val_data_loader(batch_size, val_dir, local_rank=0, world_size=1, **kwargs):
    logger.info("Get val data loader")
    val_df = pd.read_csv(os.path.join(val_dir, "val.csv"))
    val_dataset = TextDataset(val_df, rank=local_rank, world_size=world_size)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        **kwargs
    )
    return val_dataloader

# test data loader
def _get_test_data_loader(batch_size, test_dir, local_rank=0, world_size=1, **kwargs):
    logger.info("Get test data loader")
    test_df = pd.read_csv(os.path.join(test_dir, "test.csv"))
    test_dataset = TextDataset(test_df, rank=local_rank, world_size=world_size)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    return test_dataloader

# Classifier model
class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, config_fn, num_classes, state_dict:collections.OrderedDict):
        super(SimpleGPT2SequenceClassifier,self).__init__()

        config = GPTNeoXConfig.from_pretrained(config_fn)
        model = GPTNeoXModel(config)
        self.gpt2model = load_weight(model, state_dict)
        # self.gpt2model = model
        
        ##########################################################################################
        # SHIT CODE !!!!
        # I don't know how to get this input dim ... SHIT
        self.fc1 = nn.Linear(786432, num_classes)
        ##########################################################################################    
    
    """
        to Module list
    """
    def to_layers(self):
        _layers = self.gpt2model.to_layers()
        _layers.append(nn.Flatten(1, 2)) # Flatten 1 to 2 dims
        _layers.append(self.fc1)
        return _layers
        
    def forward(self, input_id, mask=None):
        """
        Args:
                input_id: encoded inputs ids of sent.
                mask: mask of the input_id, we set it with all 1.
        """
        mask=torch.ones(input_id.shape[0], 1, input_id.shape[1]).to(input_id.device)
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output

def load_stat_files(models_dir, pattern):
    ret = collections.OrderedDict([])
    for filename in os.listdir(models_dir):
        if bool(re.match(pattern, filename)):
            logger.debug(
                "Processing file : {}".format(
                    os.path.join(models_dir, filename)
                )
            )
            
            if os.path.exists(os.path.join(models_dir, filename)):
                state_slice = torch.load(os.path.join(models_dir, filename), map_location='cpu')
                ret.update(state_slice)
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(models_dir, filename))
            
            ##########################################################################################
            # ATTENTION !!!!
            # REMEMBER TO CLEAR IT
            # break
            ########################################################################################## 

    return ret

def init_model(state_dict, config_path='./zero3_config.json'):
    my_mpu = PipelineParallelGrid()
    with deepspeed.zero.Init(# data_parallel_group=my_mpu.get_data_parallel_group(),
                            mpu=my_mpu,
                            remote_device='cpu', # initial device to store weights
                            enabled=True, # if F, this context has no effect
                            pin_memory=True, # potentially increase performance
                            config_dict_or_path=config_path):

         # Get the tokenizer and model
        model_device = 'cpu'
        config_fn = "./pretrained_model/config.json"
        if os.path.exists(config_fn):
            model = SimpleGPT2SequenceClassifier(
                config_fn,
                num_classes=5,
                state_dict=state_dict).to(model_device)
        else :
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_fn)
    
    optimizer = DeepSpeedCPUAdam(model.parameters())
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config_path,
        dist_init_required=False
    )

    return model, optimizer

def get_fake_input(batch_size, seq_len):
    return torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long)

def train(args, model, optimizer, name="", ost=sys.stdout):
    train_loader = _get_train_data_loader(args.batch_size, args.train_dir, args.local_rank, args.world_size)
    test_loader = _get_test_data_loader(args.batch_size, args.test_dir, args.local_rank, args.world_size)
    val_loader = _get_val_data_loader(args.batch_size, args.val_dir, args.local_rank, args.world_size)

    train_loader = _get_train_data_loader(args.batch_size, args.train_dir)
    test_loader = _get_test_data_loader(args.batch_size, args.test_dir)
    val_loader = _get_val_data_loader(args.batch_size, args.val_dir)

    # descriptions = pd.read_csv('netflix_titles.csv')['description']
    # max_length = max([len(tokenizer.encode(description)) for description in descriptions])
    # print("Max length: {}".format(max_length))
    # dataset = NetflixDataset(descriptions, tokenizer, max_length=max_length)
    # train_size = int(0.9 * len(dataset))
    # train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)


    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm.tqdm(train_loader):
            train_label = train_label.to(torch.cuda.current_device())
            mask = train_input['attention_mask'].to(torch.cuda.current_device())
            input_id = train_input["input_ids"].squeeze(1).to(torch.cuda.current_device())
            # print(input_id.shape)
            
            model.zero_grad()

            output = model(input_id)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            model.backward(batch_loss)
            model.step()
            
            # acc = (output.argmax(dim=1)==train_label).sum().item()
            # total_acc_train += acc
            # batch_loss.backward()
            # optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        # validate model on validation data
        with torch.no_grad():
            for val_input, val_label in val_loader:
                val_label = val_label.to(torch.cuda.current_device())
                mask = val_input['attention_mask'].to(torch.cuda.current_device())
                input_id = val_input['input_ids'].squeeze(1).to(torch.cuda.current_device())
                
                output = model(input_id, mask=mask)
                
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1)==val_label).sum().item()
                total_acc_val += acc
                
            logger.info(
            f"Epochs: {epoch + 1} | Train Loss: {total_loss_train/len(train_loader): .3f} \
            | Train Accuracy: {total_acc_train / len(train_loader.dataset): .3f} \
            | Val Loss: {total_loss_val / len(val_loader.dataset): .3f} \
            | Val Accuracy: {total_acc_val / len(val_loader.dataset): .3f}")

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    logger.info("Args : " + str(args))
    deepspeed.init_distributed(verbose=False)

    pattern = re.compile("^pytorch_model-.*.bin$")
    # state_dict = load_stat_files("./pretrained_model/", pattern)
    state_dict = {}

    model, optimizer = init_model(state_dict)

    
    
    train(args, model, optimizer, name="zero3", ost=sys.stdout)

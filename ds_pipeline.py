#!/usr/bin/env python3

import os
from re import A
import sys
import time
import argparse

import collections

import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.models import vgg19

from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

from GPT2.transformer_model import GPT2Model
from GPT2.utils import load_weight
from transformers import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# labels
labels = {
    "business": 0,
    "entertainment": 1,
    "sport": 2,
    "tech": 3,
    "politics": 4
}

def get_argparser():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        metavar="N",
        help="input batch size for training (default: 2)",
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

def module2seq(model):
    blocks = []
    for name, layer in model.named_modules():
        blocks.append(layer)
    
    return nn.Sequential(
        *blocks
    )

# Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in df['text']]
        
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

# Classifier model
class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, state_dict:collections.OrderedDict):
        super(SimpleGPT2SequenceClassifier,self).__init__()

        config = GPT2Config()
        model = GPT2Model(config)
        self.gpt2model = model
        self.gpt2model = load_weight(model, state_dict)
        # self.gpt2model = GPT2Model.from_pretrained("gpt2")
        # print(self.gpt2model)
        
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)
    
    """
        to Module list
    """
    def to_layers(self):
        _layers = self.gpt2model.to_layers()
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

# train data loader
def _get_dataset(batch_size, filename, local_rank, **kwargs):
    torch.distributed.barrier()
    if local_rank != 0:
        torch.distributed.barrier()
    df = pd.read_csv(filename)
    dataset = Dataset(df)
    if local_rank == 0:
        torch.distributed.barrier()

    return dataset

'''
    Train Pipeline main function.
'''
def train_pipe(args, _state_dict, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    # CONFIG = GPT2Config(n_embd=args.hidden_dim, n_head=args.head_num, n_layer=args.layers_num, embd_pdrop=0)
    # specs = get_GPTLayers(CONFIG, 3e-4, (0.9, 0.999))
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model = GPT2Model.from_pretrained('gpt2')
    ## push it to cpu
    model_device = 'cpu'
    model = SimpleGPT2SequenceClassifier(hidden_size=args.hidden_size, \
        num_classes=5, max_seq_len=args.max_seq_len, \
        state_dict=_state_dict).to(model_device)

    specs = nn.Sequential(
        *model.to_layers()
    )

    
    net = PipelineModule(layers=specs,
                         loss_fn=nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    criterion = nn.CrossEntropyLoss()

    '''
        Here batch size is the total batch size.
    '''
    trainset = _get_dataset(1, args.train_dir + "/train.csv", args.local_rank)
    testset = _get_dataset(1, args.test_dir + "/test.csv", args.local_rank)
    validset = _get_dataset(1, args.val_dir + "/val.csv", args.local_rank)

    engine, _, dataloader, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    # dataloader = RepeatingLoader(dataloader)
    # data_iter = iter(dataloader)

    for step in range(args.epochs):
        loss = engine.train_batch()
        print(loss)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    if os.path.exists('./model/pytorch_model.bin'):
        state_dict = torch.load('./model/pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()

    train_pipe(args, state_dict)


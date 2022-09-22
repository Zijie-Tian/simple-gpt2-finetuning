import argparse
import json
import logging
import os
import sys
import collections

#import sagemaker_containers
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np

from torch import nn
from torch.optim import Adam
from transformers import GPT2Tokenizer
from tqdm import tqdm

from GPT2.transformer_model import GPT2Model
from GPT2.utils import load_weight
from transformers import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

# Initialize the logger and set the level to info
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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
    parser.add_argument(
        "--lr", type=float, default=1e-5, metavar="LR", help="learning rate (default: 1e-5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")

    parser.add_argument("--hidden-size", type=int, default=768, metavar="HS", help="hidden size (default: 768)")
    parser.add_argument("--max-seq-len", type=int, default=128, metavar="MSL", help="max sequence length (default: 128)")
    
    # Container environment
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])

    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--train-dir", type=str)
    parser.add_argument("--val-dir", type=str)
    parser.add_argument("--test-dir", type=str)
    parser.add_argument("--num-gpus", type=int)

    return parser

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
        # self.gpt2model = model
        # self.gpt2model = load_weight(model, state_dict)
        self.gpt2model = GPT2Model.from_pretrained("gpt2")

        # print(self.gpt2model)
        
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)
        
    def forward(self, input_id, mask=None):
        """
        Args:
                input_id: encoded inputs ids of sent.
                mask: mask of the input_id, we set it with all 1.
        """
        mask = torch.ones(input_id.shape[0], 1, input_id.shape[1]).to(input_id.device)
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output

# train data loader
def _get_train_data_loader(batch_size, train_dir, **kwargs):
    train_df = pd.read_csv(os.path.join(train_dir, "train.csv"))
    train_dataset = Dataset(train_df)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        **kwargs
    )
    return train_dataloader

# val data loader
def _get_val_data_loader(batch_size, val_dir, **kwargs):
    logger.info("Get val data loader")
    val_df = pd.read_csv(os.path.join(val_dir, "val.csv"))
    val_dataset = Dataset(val_df)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        **kwargs
    )
    return val_dataloader

# test data loader
def _get_test_data_loader(batch_size, test_dir, **kwargs):
    logger.info("Get test data loader")
    test_df = pd.read_csv(os.path.join(test_dir, "test.csv"))
    test_dataset = Dataset(test_df)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    return test_dataloader

# train
def train(args, _state_dict):
    # set up GPU training (if using GPU)
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # load train, validation and test data
    train_loader = _get_train_data_loader(args.batch_size, args.train_dir, **kwargs)
    val_loader = _get_val_data_loader(args.batch_size, args.val_dir, **kwargs)
    test_loader = _get_test_data_loader(args.batch_size, args.test_dir, **kwargs)

    # print logging info
    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of val data".format(
            len(val_loader.sampler),
            len(val_loader.dataset),
            100.0 * len(val_loader.sampler) / len(val_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    # initialize model and parameters
    model = SimpleGPT2SequenceClassifier(hidden_size=args.hidden_size, num_classes=5, max_seq_len=args.max_seq_len, state_dict=_state_dict).to(device)
    EPOCHS = args.epochs
    LR = args.lr

    # use cross-entropy as the loss function
    criterion = nn.CrossEntropyLoss()

    # use Adam as the optimizer
    optimizer = Adam(model.parameters(), lr=LR)

    # enable GPU training (if using GPU)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # training loop
    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_loader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            # print(input_id.shape)
            
            model.zero_grad()

            output = model(input_id)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1)==train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        # validate model on validation data
        with torch.no_grad():
            for val_input, val_label in val_loader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                
                output = model(input_id, mask=mask)
                
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1)==val_label).sum().item()
                total_acc_val += acc
                
            logger.info(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_loader): .3f} \
            | Train Accuracy: {total_acc_train / len(train_loader.dataset): .3f} \
            | Val Loss: {total_loss_val / len(val_loader.dataset): .3f} \
            | Val Accuracy: {total_acc_val / len(val_loader.dataset): .3f}")
    
    # evaluate model performance on unseen data
    # test(model, test_loader, device)
    
    # save model
    # save_model(model, args.model_dir)


if __name__ == '__main__':
    parser = get_argparser()

    args = parser.parse_args()

    if os.path.exists('./model/pytorch_model.bin'):
        state_dict = torch.load('./model/pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()
    
    train(args, state_dict)






import numpy as np
import pandas as pd
import torch
import os
import random
import math
import torch.nn as nn
import transformers

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup

import argparse


# In[2]:

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("input", help = "Path to Input File in CSV Format")
parser.add_argument("model_path", help = "Path to Trained Model Tensors")
 
# Read arguments from command line
args = parser.parse_args()
 
filename = args.input
model_path = args.model_path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

"""# Import BERT Model and BERT Tokenizer"""

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

df = pd.read_csv(filename)

text = df['input']

max_seq = 128

# tokenize and encode sequences in the test set
test_text = tokenizer.batch_encode_plus(text.tolist(), padding='max_length', max_length = max_seq, truncation = True,  return_token_type_ids=False)

# for test set
test_seq = torch.tensor(test_text['input_ids'])
test_mask = torch.tensor(test_text['attention_mask'])

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#define a batch size
batch_size = 8

# wrap tensors
test_data = TensorDataset(test_seq, test_mask)

# dataLoader for train set
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# push the model to GPU
model = model.to(device)

model.load_state_dict(torch.load(model_path))


model.eval()
predictions=np.empty([0])
for step, batch in enumerate(test_dataloader):
    
    # progress update after every 50 batches.
    if step % 20 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask = batch
    output = model(sent_id, mask)[0]
    output = torch.argmax(output, axis = 1)
    output = output.detach().cpu().numpy()
    predictions = np.concatenate((predictions, output))

df['labels'] = predictions
df.to_csv(filename, index = None)

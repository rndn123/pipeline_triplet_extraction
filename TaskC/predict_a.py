import numpy as np
import pandas as pd
import torch
import transformers
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse

# Set the seed value all over the place to make this reproducible.
seed_val = 100

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


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

df_test = pd.read_csv(filename)
df_test = df_test[(df_test['sub_num'] < df_test['pred_num']) & (df_test['pred_num'] < df_test['obj_num'])]

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

testing_text = df_test['input']

max_seq = 128
# tokenize and encode sequences in the test set
test_text = tokenizer.batch_encode_plus(testing_text.tolist(), padding='max_length', max_length = max_seq, truncation = True,  return_token_type_ids=False)

# for test set
test_seq = torch.tensor(test_text['input_ids'])
test_mask = torch.tensor(test_text['attention_mask'])

"""# Create DataLoaders"""

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#define a batch size
batch_size = 8

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()
# Load Pre-Trained Model
model.load_state_dict(torch.load(model_path))

"""#Testing"""


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#define a batch size
batch_size = 8

# wrap tensors
test_data = TensorDataset(test_seq, test_mask)

# dataLoader for train set
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model.eval()
test_predictions=np.empty([0])
for step, batch in enumerate(test_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask = batch
    output = model(sent_id, mask)[0]
    output = torch.argmax(output, axis = 1)
    output = output.detach().cpu().numpy()
    test_predictions = np.concatenate((test_predictions,output))

df_test['labels'] = test_predictions
df_test.to_csv(filename, index = None)

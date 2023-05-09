import numpy as np
import pandas as pd
import torch
import transformers
import matplotlib.pyplot as plt
import random
import numpy as np

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

df_train = pd.read_csv("../Preprocessed_Dataset/Training_Triplets_C.csv")
df_val = pd.read_csv("../Preprocessed_Dataset/Trial_Triplets_C.csv")
df_test = pd.read_csv('../Preprocessed_Dataset/Test_Triplets_C.csv')

df_train = df_train[(df_train['end_pred'] < df_train['start_obj'])]
df_val = df_val[(df_val['end_pred'] < df_val['start_obj'])]
df_test = df_test[(df_test['end_pred'] < df_test['start_obj'])]

#df_train = df_train[(df_train['sub_num'] + 2 >= df_train['obj_num'])]
#df_val = df_val[(df_val['sub_num'] + 2 >= df_val['obj_num'])]


#df_train = df_train[(df_train['sub_num'] < df_train['pred_num']) & (df_train['pred_num'] < df_train['obj_num'])]
#df_val = df_val[(df_val['sub_num'] < df_val['pred_num']) & (df_val['pred_num'] < df_val['obj_num'])]

print(f"Number of Training Examples = {len(df_train.index)}")
print(f"Number of Valid Examples = {len(df_val.index)}")

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')


training_text = df_train['input']
validating_text = df_val['input']
testing_text = df_test['input']
training_labels = df_train['labels']
validating_labels = df_val['labels']
testing_labels = df_test['labels']

max_seq = 128

# tokenize and encode sequences in the training set
train_text = tokenizer.batch_encode_plus(training_text.tolist(), padding='max_length', max_length = max_seq, truncation = True,  return_token_type_ids=False)
# tokenize and encode sequences in the val set
val_text = tokenizer.batch_encode_plus(validating_text.tolist(), padding='max_length', max_length = max_seq, truncation = True, return_token_type_ids=False)
# tokenize and encode sequences in the test set
test_text = tokenizer.batch_encode_plus(testing_text.tolist(), padding='max_length', max_length = max_seq, truncation = True,  return_token_type_ids=False)

# for train set
train_seq = torch.tensor(train_text['input_ids'])
train_mask = torch.tensor(train_text['attention_mask'])
train_y = torch.tensor(training_labels.tolist())

# for val set
val_seq = torch.tensor(val_text['input_ids'])
val_mask = torch.tensor(val_text['attention_mask'])
val_y = torch.tensor(validating_labels.tolist())

# for test set
test_seq = torch.tensor(test_text['input_ids'])
test_mask = torch.tensor(test_text['attention_mask'])
test_y = torch.tensor(testing_labels.tolist())

"""# Create DataLoaders"""

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#define a batch size
batch_size = 8

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# dataLoader for train set
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle = True)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# dataLoader for train set
validation_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle = True)

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

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(), lr = 1e-5)


# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 10
# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps = 500, num_training_steps = total_steps, power = 0.5)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def accuracy(predict, targets):
  n_correct = torch.sum((predict==targets) * (targets==1)).item()
  return n_correct

def calculate_labels(targets):
  return torch.sum(targets==1).item()

"""Helper function for formatting elapsed times as `hh:mm:ss`

"""

torch.save(model.state_dict(), 'scibert_triplets_C.pt')

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

"""#Training Cell"""

best_valid_loss = float('inf')
best_valid_F1 = 0
early_stopping_limit = 0
# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels,return_dict=True)

        loss = result.loss
        logits = result.logits

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()
    
    

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    n_correct = 0
    n_predict = 0
    n_ground = 0

    # Evaluate data for one epoch
    for step, batch in enumerate(validation_dataloader):
    
    # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(validation_dataloader), elapsed))

        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        # Get the loss and "logits" output by the model. The "logits" are the 
        # output values prior to applying an activation function like the 
        # softmax.
        loss = result.loss
        logits = result.logits
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        #logits = logits.detach().cpu().numpy()
        #label_ids = b_labels.to('cpu').numpy()

        predict = torch.argmax(logits, axis = 1)
        n_correct += accuracy(predict, b_labels)
        n_predict += calculate_labels(predict)
        n_ground += calculate_labels(b_labels)

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        # total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    #avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    #print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    P = n_correct/n_predict if n_predict else 0.0
    R = n_correct/n_ground if n_ground else 0.0
    F1 = (2*P*R)/(P+R) if P+R else 0.0
    print(f"  Validation True Positive  = {n_ground}")
    print("  Validation Precision: {0:.4f}".format(P))
    print("  Validation Recall: {0:.4f}".format(R))
    print("  Validation F1 Score: {0:.4f}".format(F1))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid Precision' : P,
            'Valid Recall' : R,
            'Valid. F1': F1,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    if best_valid_F1 < F1:
        best_valid_F1 = F1
        torch.save(model.state_dict(), 'scibert_triplets_C.pt')
        early_stopping_limit = 0
    else:
        early_stopping_limit += 1
    
    
    if(early_stopping_limit == 3):
        print("Early Stopping Patience Limit reached. Terminating Training")
        break

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

"""Let's view the summary of the training process."""

# Display floats with two decimal places.
pd.set_option('precision', 4)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# Display the table.
df_stats.to_csv("Training_stats_C.csv")

model.load_state_dict(torch.load('scibert_triplets_C.pt'))

"""#Testing"""

torch.cuda.empty_cache()

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#define a batch size
batch_size = 8

# wrap tensors
test_data = TensorDataset(test_seq, test_mask)

# dataLoader for train set
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model.eval()
test_predictions=np.empty([0,2])
for step, batch in enumerate(test_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask = batch
    preds = model(sent_id, mask)
    test_predictions = np.concatenate((test_predictions,preds[0].detach().cpu().numpy()))

test_predictions = np.argmax(test_predictions, axis=1)

print(accuracy_score(test_predictions, testing_labels)*100)
print(confusion_matrix(testing_labels,test_predictions))
print(classification_report(testing_labels, test_predictions))

df_test['predict'] = test_predictions
df_test.to_csv("test_results_C.csv", index = None)

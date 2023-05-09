import pandas as pd
import numpy as np

import joblib
import torch
#import wandb

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel

import pandas as pd
from ast import literal_eval as load
import nltk
from tqdm import tqdm

def process(sentences, tokens, predicted):
    predicted_tag = []
    for i in range(len(sentences)):
    	pred = predicted[i]
    	sent = sentences[i]
    	token = tokens[i]
    	tags = []
    	k = 0
    	word = ''
    	tag = ''
    	for j in range(1, len(token)-1):
    		s = token[j]
    		if(word == ''):
    			tag = pred[j]
    		if(s.startswith("##")):
    			word += s[2:]
    		else:
    			word += s
    		if(word == sent[k].lower()):
    			k = k + 1
    			word = ''
    			tags.append(tag)
    	predicted_tag.append(tags)
    return predicted_tag
    
def eval_fn(data_loader, model, device):
    model.eval()
    tokenized, predicted = [], []
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        input_ids = data['ids'][0]
        _, predicted_tag = model(x = 0, **data)
        predicted_tag = predicted_tag[0]
        tokens = config.TOKENIZER.convert_ids_to_tokens(input_ids)
        predicted.append(predicted_tag)
        tokenized.append(tokens)
    return tokenized, predicted

def process_data(data_path):
    
    df = pd.read_csv(data_path, encoding="latin-1")
    sentences, pos, tag = [], [], []
    
    for i in range(len(df)):
        
        sent = load(df.iloc[i, 5])
        pos_tag = [2] * len(sent)
        labels = [0] * len(sent)
        sentences.append(sent)
        pos.append(pos_tag)
        tag.append(labels)
        
    return sentences, pos, tag

if __name__ == "__main__":

    sentences, pos, tag = process_data(config.TESTING_FILE)
    
    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]
    
    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))
    
    test_dataset = dataset.EntityDataset(
        texts=sentences, pos=pos, tags=tag, pad = False
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=4
    )
    
    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load('scibert_crf_f1.pt'))
    model.to(device)
    tokenized, predicted = eval_fn(test_data_loader, model, device)
    predicted_tag = process(sentences, tokenized, predicted)
    
    df = pd.read_csv(config.TESTING_FILE)
    df['predicted_tag'] = predicted_tag
    df.to_csv("test_results_159.csv", index = None)

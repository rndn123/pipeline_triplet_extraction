import torch
import numpy
import config
from tqdm import tqdm

def prediction(tokens, predicted_tag, target_tag):
	
	tp, pred, total = 0, 0, 0
	flag, tf = 0, 0
	length = len(tokens)
	for i in range(1,length-1):
		if(tokens[i] == '[SEP]'):
			break
		if(tokens[i].startswith('##')):
			continue
		if (predicted_tag[i] == 2 and target_tag[i] == 2):
			tp = tp + (flag & tf)
			flag, tf = 0, 0
		else:
			if (predicted_tag[i] == 2):
				flag = 0
			if (target_tag[i] == 2):
				tf = 0
		if (predicted_tag[i] == 0 and target_tag[i] == 0):
			tp = tp + (flag & tf)
			flag, tf = 1, 1
			pred = pred + 1
			total = total + 1
		elif (predicted_tag[i] == 1 and predicted_tag[i-1] == 2 and target_tag[i] == 0):
			flag, tf = 1, 1
			pred = pred + 1
			total = total + 1
		else:
			if (predicted_tag[i] == 0):
				flag = 1
				pred = pred + 1
			if (target_tag[i] == 0):
				tf = 1
				total = total + 1
	return tp, pred, total

def train_scaffold(data_loader, model, optimizer, device, scheduler, wandb_config):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss, tag = model(x = 1, **data)
        loss = wandb_config.lambda1 * loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)
    
def train_tdmsci(data_loader, model, optimizer, device, scheduler, wandb_config):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss, tag = model(x = 0, **data)
        loss = wandb_config.lambda2 * loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)    

def train_fn(data_loader, model, optimizer, device, scheduler, wandb_config):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss, tag = model(x = 0, **data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)

def eval_fn(data_loader, model, device):
    model.eval()
    n_correct, n_predict, n_ground = 0,0,0
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss, predicted_tag = model(x = 0, **data)
        target_tag = data['target_tag'][0]
        input_ids = data['ids'][0]
        predicted_tag = predicted_tag[0]
        tokens = config.TOKENIZER.convert_ids_to_tokens(input_ids)
        n_c, n_p, n_g = prediction(tokens, predicted_tag, target_tag)
        n_correct += n_c
        n_predict += n_p
        n_ground += n_g
        final_loss += loss.item()
    precision = n_correct/n_predict if n_predict else 0
    recall = n_correct/n_ground if n_correct else 0
    F1 = 2*precision*recall/(precision+recall) if (precision + recall) else 0
    return final_loss / len(data_loader), precision, recall, F1

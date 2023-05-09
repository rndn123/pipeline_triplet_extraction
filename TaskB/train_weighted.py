import pandas as pd
import numpy as np

import joblib
import torch
import random
import os

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel

random.seed(config.seed_val)
np.random.seed(config.seed_val)
torch.manual_seed(config.seed_val)
torch.cuda.manual_seed_all(config.seed_val)

def process_data(df):
    
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    
    return sentences, pos, tag

enc_pos = preprocessing.LabelEncoder()
enc_tag = preprocessing.LabelEncoder()

'''df_train = pd.read_csv("NCG_Train.csv")
df_train.columns = ['Sentence #', "Word", "Tag", "POS"]

df_train.loc[:, "POS"] = enc_pos.fit_transform(df_train["POS"])
df_train.loc[:, "Tag"] = enc_tag.fit_transform(df_train["Tag"])

df_valid = pd.read_csv("NCG_Valid.csv")
df_valid.columns = ['Sentence #', "Word", "Tag", "POS"]

df_valid.loc[:, "POS"] = enc_pos.transform(df_valid["POS"])
df_valid.loc[:, "Tag"] = enc_tag.transform(df_valid["Tag"])'''

df = pd.read_csv("NCG_phrases.csv")
df.columns = ['Sentence #', "Word", "Tag", "POS"]

df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

df_scierc = pd.read_csv("scierc_sent_phrases.csv")
df_scierc.columns = ['Sentence #', "Word", "Tag", "POS"]

df_scierc.loc[:, "POS"] = enc_pos.transform(df_scierc["POS"])
df_scierc.loc[:, "Tag"] = enc_tag.transform(df_scierc["Tag"])

df_sciclaim = pd.read_csv("SciClaim.csv")
df_sciclaim.columns = ['Sentence #', "Word", "Tag", "POS"]

df_sciclaim.loc[:, "POS"] = enc_pos.fit_transform(df_sciclaim["POS"])
df_sciclaim.loc[:, "Tag"] = enc_tag.fit_transform(df_sciclaim["Tag"])

df_tdm = pd.read_csv("TDMSci_dataset.csv")
df_tdm.columns = ['Sentence #', "Word", "Tag", "POS"]

df_tdm.loc[:, "POS"] = enc_pos.fit_transform(df_tdm["POS"])
df_tdm.loc[:, "Tag"] = enc_tag.fit_transform(df_tdm["Tag"])


#train_sentences, train_pos, train_tag = process_data(df_train)
#valid_sentences, valid_pos, valid_tag = process_data(df_valid)
sentences, pos, tag = process_data(df)

train_sentences, valid_sentences, train_pos, valid_pos, train_tag, valid_tag = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1)

scierc_sentences, scierc_pos, scierc_tag = process_data(df_scierc)

sciclaim_sentences, sciclaim_pos, sciclaim_tag = process_data(df_sciclaim)

tdm_sentences, tdm_pos, tdm_tag = process_data(df_tdm)

meta_data = {
"enc_pos": enc_pos,
"enc_tag": enc_tag,
}

pad = True if config.TRAIN_BATCH_SIZE > 1 else False

joblib.dump(meta_data, "meta.bin")

num_pos = len(list(enc_pos.classes_))
num_tag = len(list(enc_tag.classes_))

print(num_pos, num_tag)

train_dataset = dataset.EntityDataset(
texts=train_sentences, pos=train_pos, tags=train_tag, pad = pad
)

train_data_loader = torch.utils.data.DataLoader(
train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=8, shuffle = True
)

valid_dataset = dataset.EntityDataset(
texts=valid_sentences, pos=valid_pos, tags=valid_tag, pad = False
)

valid_data_loader = torch.utils.data.DataLoader(
valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
)


scierc_dataset = dataset.EntityDataset(
texts=scierc_sentences, pos=scierc_pos, tags=scierc_tag, pad = pad
)

scierc_data_loader = torch.utils.data.DataLoader(
scierc_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=8, shuffle = True
)

sciclaim_dataset = dataset.EntityDataset(
texts=sciclaim_sentences, pos=sciclaim_pos, tags=sciclaim_tag, pad = pad
)

sciclaim_data_loader = torch.utils.data.DataLoader(
sciclaim_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=8, shuffle = True
)


tdm_dataset = dataset.EntityDataset(
texts=tdm_sentences, pos=tdm_pos, tags=tdm_tag, pad = pad
)

tdmsci_data_loader = torch.utils.data.DataLoader(
tdm_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=8, shuffle = True
)



'''param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
{
    "params": [
        p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
    ],
    "weight_decay": 0.001,
},
{
    "params": [
        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
    ],
    "weight_decay": 0.0,
},
]'''

def training():

	config_defaults = {
		'dropout' : 0.1,
		'learning_bert_rate' : 1e-5,
		'learning_rate': 1e-4,
		'lambda1' : 0.1
	}
	
	# Initialize a new wandb run
	wandb.init(config=config_defaults, project = "SCIERC")
	# Config is a variable that holds and saves hyperparameters and inputs
	wandb_config = wandb.config
	
	device = torch.device("cuda")
	model = EntityModel(wandb_config = wandb_config, num_tag=num_tag, num_pos=num_pos)
	model.to(device)
	#torch.save(model.state_dict(), "initializer_0.pt")
	model.load_state_dict(torch.load("initializer.pt"))
			
	total = len(train_sentences) + len(scierc_sentences)
	num_train_steps = int(total/ config.TRAIN_BATCH_SIZE * config.EPOCHS)
	
	pretrained = model.bert.parameters()
	pretrained_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]
	new_params= [v for k, v in model.named_parameters() if k not in pretrained_names]
	optimizer = AdamW([{'params': pretrained}, {'params': new_params, 'lr': wandb_config.learning_rate}],lr=wandb_config.learning_bert_rate)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
	
	best_loss = np.inf
	best_val_F1 = 0
	
	for epoch in range(config.EPOCHS):
	
		scierc_loss = engine.train_scaffold(scierc_data_loader, model, optimizer, device, scheduler, wandb_config)
		#tdmsci_loss = engine.train_fn(tdmsci_data_loader, model, optimizer, device, scheduler, wandb_config)
		sciclaim_loss = engine.train_scaffold(sciclaim_data_loader, model, optimizer, device, scheduler, wandb_config)
		train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, wandb_config)
		valid_loss, precision, recall, F1 = engine.eval_fn(valid_data_loader, model, device)
		print(f"SCICLAIM Loss = {sciclaim_loss}, Train Loss = {train_loss}, Valid Loss = {valid_loss}")
		print((f'Validation Results - Precision: {precision}, Recall: {recall}, F1 Score: {F1}'))
		
		#Saving the best model
		if(best_val_F1 < F1):
			best_val_F1 = F1
			torch.save(model.state_dict(), os.path.join(wandb.run.dir, "scibert_crf_f1.pt"))
		
		wandb.log({"Training Loss":train_loss})
		wandb.log({"Validation Loss":valid_loss})
		#wandb.log({"SCIERC Loss":scierc_loss})
		#wandb.log({"SCICLAIM Loss":sciclaim_loss})
		#wandb.log({"TDMSCI Loss":scierc_loss})
		wandb.log({"Validation Precision":precision})
		wandb.log({"Validation Recall":recall})
		wandb.log({"Validation F1 Score":F1})
		
	#torch.save(model.state_dict(), os.path.join(wandb.run.dir, "scibert_crf_5.pt"))

import wandb
wandb.login()

sweep_config = {
    'method': 'grid',
    'metric': {
      'name': 'Validation F1 Score',
      'goal': 'maximize'   
    },
    'parameters': {
        'dropout': {
            'values': [0.1, 0.2]
        },
        'learning_bert_rate': {
            "values" : [1e-5, 2e-5]
        },
        'learning_rate': {
            "values" : [1e-4, 2e-4]
        },
        'lambda1' : {
            "values" : [0.1, 0.2]
        },
    }
}

#training()

sweep_id = wandb.sweep(sweep_config, project="SCICLAIM")
wandb.agent(sweep_id, training)

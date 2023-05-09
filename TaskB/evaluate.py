import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.columns = ['Sentence #', "Word", "Tag", "POS"]
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]
    
    df.loc[:, "POS"] = enc_pos.transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.transform(df["Tag"])
    
    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    
    print(len(sentences))
    return sentences, pos, tag, enc_pos, enc_tag


if __name__ == "__main__":
    sentences, pos, tag, enc_pos, enc_tag = process_data(config.TESTING_FILE)

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    test_dataset = dataset.EntityDataset(
        texts=sentences, pos=pos, tags=tag
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    test_loss, precision, recall, F1 = engine.eval_fn(test_data_loader, model, device)
    print(f"Test Loss = {test_loss}, Precision = {precision}, Recall = {recall}, F1 Score = {F1}")

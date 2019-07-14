import pandas as pd 
import numpy as np 
import os
import click

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader 
from apex import amp # has no apex yet , todo -> install apex on windows)
from tqdm import tqdm 

from config import Config # add Config module
from models import BertForTokenClassificationMultiOutput #add this model
from pytorch_pretrained_bert import BertAdam # install pytorch_pretrained_bert
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils import *
import gc

device = torch.device('cuda')

def custom_loss_BCE(data, target, loss_weight):
    bce_loss1 = nn.BCEWithLogitsLoss(weight=target[:,1:2])(data[:,:1], target[:,:1])
    bce_loss2 = nn.BCEWithLogitsLoss()(data[:,1:], target[:,2:])
    return (bce_loss1 * loss_weight) + bce_loss2

def train(model, optimizer, loader, criterion):
    avg_loss = 0
    avg_accuracy = 0
    lossf = None
    tk0 = tqdm(enumerate(loader), total = len(loader), leave=False)
    for i, (x_batch, added_fts, y_batch) in tk0:
        optimizer.zero_grad()
        all_y_pred = model(
            x_batch.to(device),
            f=added_fts.to(device),
            attention_mask=(x_batch > 0).to(device),
            labels=None
        )
        y_pred = all_y_pred[:, 0]
        loss = criterion(all_y_pred, y_batch.to(device))
        loss =/ Config.accumulation_steps

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (i+1) % Config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98 * lossf + 0.02 * loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss=lossf)

        avg_loss += loss.item()/ len(loader)

        avg_accuracy += torch.mean(((torch.sigmoid(y_pred) > 0.5) == (y_batch[:, 0] > 0.5).to(device)).to(torch.float)).item() / len(loader)

        return avg_loss, avg_accuracy

    def valid(model, loader, valid_df):
        model.eval()
        valid_preds =[]
        tk0 = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i, (x_batch, added_fts) in enumerate(tk0):
                pred = model(x_batch.to(device),
                 f=added_fts.to(device),
                 criterion)

        
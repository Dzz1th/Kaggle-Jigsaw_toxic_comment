import numpy as np 
import pandas as pd 
import os
import time
import gc 
import random
from tqdm import tqdm 
from keras.preprocessing import text , sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F 
from torch import tensor

max_features = 10000

NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 220

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

## upload embedding matrix from text_preprocessing

def sigmoid(x):
    return 1 / 1 + np.exp(-x)

def train_model(model, train, test, loss_fn, output_dim, lr=0.001,
                batch_size = 512, n_epoch = 4,
                enable_checkpoint_ensemble = True):
    param_lrs = [{'params':param , 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs , lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer , lambda epoch: 0.6**epoch)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test , batch_size = batch_size, shuffle=True)
    all_test_predict = []
    checkpoint_weights = [2** epoch for epoch in range(n_epoch)]

    for epoch in range(n_epoch):
        start_time = time.time()

        scheduler.step()

        model.train()
        avg_loss = 0

        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)
            loss = loss_fn(y_pred , y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        test_preds = np.zeros((len(test) , output_dim))

        for i , x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())
            test_preds[i*batch_size:(i+1)*batch_size,:] = y_pred

            all_test_predict.append(test_preds)
            ellapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t time = {:.2f}s'.format(
                epoch+1, n_epoch , avg_loss, ellapsed_time))

    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_predict, weights = checkpoint_weights , axis=0)
    else:
        test_preds = all_test_predict[-1]

    return test_preds

class CustomLoss(nn.Module):
    def __init__(self, loss_weight):
        super(CustomLoss, self).__init__()
        self.loss_weight = loss_weight
        self.bin_loss = nn.BCEWithLogitsLoss
        self.aux_loss = nn.BCEWithLogitsLoss

    def forward(self, output_bin, target_bin, weight_bin, output_aux, target_aux):
        bin_loss = self.bin_loss(weight=weight_bin)(output_bin, target_bin)
        aux_loss = self.aux_loss()(output_aux, target_aux)
        return 0.5 * bin_loss + 0.5 * aux_loss

class SpatialDropout(nn.Dropout2d):
    def forward(self ,x):
        #x - torch.Tensor
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout , self).forward(x)
        x = x.permute(0 , 3, 2, 1)
        x = x.squeeze(2)
        return x

class SimpleLSTM(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(SimpleLSTM , self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight= nn.Parameter(torch.tensor(embedding_matrix, dtype= torch.float32))
        self.embedding.weight.requires_grad=False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size ,  LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2 , LSTM_UNITS , bidirectional=True, batch_first = True)

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        avg_pool = torch.mean(h_lstm2, 1)

        max_pool = torch.max(h_lstm2, 1)

        h_conc = torch.cat((avg_pool , max_pool) , 1)

        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linerar2 = F.relu(self.linear2(h_conc))

        hidden = h_conc_linear1 + h_conc + h_conc_linerar2
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result] , 1)

        return out


for model_idx in range(NUM_MODELS):
    print('Model' , model_idx)
    seed_everything(1234 + model_idx)

    model = SimpleLSTM(embedding_matrix, y_aux_train.shape[-1])
    model.cuda()

    test_pred = train_model(model, train_dataset, test_dataset, output_dim = y_train_torch.shape[-1],
    loss_fn= CustomLoss)

    all_test_predict.append(test_pred)
    torch.save(model.state_dict(), './{}'.format(model_idx))
    print()


        

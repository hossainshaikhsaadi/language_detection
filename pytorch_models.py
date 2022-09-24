import torch
import torch.nn as nn
import random
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import DistilBertModel, DistilBertConfig


# setting the seeds for torch, random and numpy for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class BiLSTM(nn.Module):
    # class for the BiLSTM based model
    def __init__(self, emb_dim, vocab_size, hidden_dim, num_classes):
        super(BiLSTM, self).__init__()

        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.bilstm = nn.LSTM(emb_dim, hidden_dim, num_layers = 1, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self,data):
        emb = self.embeddings(data)
        output, _ = self.bilstm(emb)
        output = self.linear(output[:,-1,:])
        return output

class Bert(nn.Module):
    # class for the BERT based model
    def __init__(self, num_classes):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.linear = nn.Linear(768, num_classes)

    def forward(self, data):
        model_output = self.model(**data)
        output = model_output.last_hidden_state[:,0,:]
        output = self.linear(output)
        return output


class Dbert(nn.Module):
    # class for DistilBERT based model
    def __init__(self, num_classes):
        super(Dbert, self).__init__()
        self.model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, data):
        model_output = self.model(**data)
        output = model_output.last_hidden_state[:,0,:]
        output = self.linear(output)
        return output







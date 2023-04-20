import os

import torch
import torch.nn as nn


MODEL_DIR = './model'

def save_weights(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"weights.{epoch}.pt"))

def load_weights(epoch, model):
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"weights.{epoch}.pt")))
    model.eval()
    return model

class CharLSTM(nn.Module):
    def __init__(self, n_hidden = 256, n_layers = 3, drop_prob = 0.2, vocab_size = 0):
        super(CharLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        
        self.emb = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(512, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, x, h):
        embedded = self.emb(x)
        out, hn = self.lstm(embedded, h)
        out = self.dropout(out)
        out = out.reshape(-1, self.n_hidden)
        out = self.fc(out)
        
        return out, hn
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)

        return (h0,c0)

# if __name__ == '__main__':
#     model = CharLSTM()
#     summary(model, (1, 512))

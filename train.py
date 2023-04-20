import os
import json
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn

from model import CharLSTM, save_weights

warnings.filterwarnings("ignore")


DATA_DIR = './data'
LOG_DIR = './logs'

BATCH_SIZE = 16
SEQ_LENGTH = 64

class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss\n')

    def add_entry(self, loss):
        self.epochs += 1
        s = '{},{}\n'.format(self.epochs, loss)
        with open(self.file, 'a') as f:
            f.write(s)

def read_batches(T, vocab_size):
    NUM_CHAR_EACH_ROW = T.shape[0]//BATCH_SIZE
    
    for start in range(0, NUM_CHAR_EACH_ROW - SEQ_LENGTH, SEQ_LENGTH):
        X = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int32)
        Y = torch.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))
        for row in range(BATCH_SIZE):
            for col in range(SEQ_LENGTH):
                X[row, col] = T[row*NUM_CHAR_EACH_ROW + start + col]
                Y[row, col, T[row*NUM_CHAR_EACH_ROW + start + col + 1]] = 1
        yield X , Y

def train(text, epochs=100, save_freq=10):

    # character to index and vice-versa mappings
    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
    print("Number of unique characters: " + str(len(char_to_idx)))

    with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)

    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model_architecture
    model = CharLSTM(vocab_size = vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()


    #Train data generation
    T = torch.tensor([char_to_idx[c] for c in text], dtype=torch.int32) #convert complete text into numerical indices

    print("Length of text:" + str(T.size)) 

    steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH  

    log = TrainLogger('training_log.csv')

    model.train()

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        epoch_loss = 0
        h = model.init_hidden(BATCH_SIZE, device)
        for idx, (X,Y) in enumerate(read_batches(T, vocab_size)):
            x = X.to(device)
            y = Y.view(-1, 86).to(device)
            
            outputs, h = model(x, h)
            loss = criterion(outputs, y)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            h = (h[0].detach(), h[1].detach())
            
            epoch_loss += loss.item()
            
            print(f"Batch {idx+1}: loss = {loss.item()}")
        
        log.add_entry(epoch_loss)
        if (epoch+1) % save_freq == 0:
            save_weights(epoch+1, model)
            print('Saved checkpoint to', f'weights.{epoch+1}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='input.txt', help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    train(open(os.path.join(DATA_DIR, args.input)).read(), args.epochs, args.freq)

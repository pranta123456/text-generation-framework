import argparse
import os
import json

import torch
import torch.nn as nn
import numpy as np

from model import CharLSTM, load_weights

DATA_DIR = './data'
MODEL_DIR = './model'

def sample(epoch, header, num_chars):
    with open(os.path.join(DATA_DIR, 'char_to_idx.json')) as f:
        char_to_idx = json.load(f)
    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharLSTM(vocab_size = vocab_size).to(device)
    model = load_weights(epoch, model)

    sampled = [char_to_idx[c] for c in header]
    print(sampled)
    

    h = model.init_hidden(1, device)
    for i in range(num_chars):
        batch = torch.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            batch[0, 0] = torch.randint(0, vocab_size, (1,))
        batch = batch.to(torch.int32)
        x = batch.to(device)
        result, h = model(x, h)
        p = nn.Softmax(dim=1)(result)
        sample = p.multinomial(num_samples=1, replacement=True)
        sampled.append(sample.item())

    return ''.join(idx_to_char[c] for c in sampled)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample some text from the trained model.')
    parser.add_argument('epoch', type=int, help='epoch checkpoint to sample from')
    parser.add_argument('--seed', default='', help='initial seed for the generated text')
    parser.add_argument('--len', type=int, default=512, help='number of characters to sample (default 512)')
    args = parser.parse_args()

    print(sample(args.epoch, args.seed, args.len))

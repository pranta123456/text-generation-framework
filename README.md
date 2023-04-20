# Text Generation

This is a text generation framework for training various model and sampling from text.

### Supported Models

* Statefull LSTM
* GPT [Coming]

### Requirements

This code is written in Python 3, and it requires the PyTorch deep learning library.

### Usage

All input data should be placed in the `data/` directory.

To train the model with default settings:
```bash
$ python train.py
```

To sample the model:
```bash
$ python sample.py 100 
```

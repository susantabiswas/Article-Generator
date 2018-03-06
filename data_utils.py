# modules for data loading and data preparation
import numpy as np


def load_dataset(data_path):
    with open(data_path, encoding='utf-8') as file:
        text_data = file.read().lower()

    # vocabulary character tokens
    vocab_chars = sorted(list(set(text_data)))

    return vocab_chars, text_data


# creates char to index mapping and reverse mapping
def create_mapping(vocab_chars):
    # mapping from char to numerical index
    char_idx = dict((char, idx) for idx, char in enumerate(vocab_chars))
    idx_char = dict((idx, char) for idx, char in enumerate(vocab_chars))

    return char_idx, idx_char


# tokenize the sentences: we split the 
def tokenize_split(text, Tx):
    # we split the input data such that for training we always feed the network a fixed sentence of
    # 40 characters and for that we make the next character as the output
    # i.e for every 40 characters long sentence we have an output character 
    
    # Tx: input timesteps
    
    # decides the difference in position in 1st characters of two consecutive input sentences 
    step = 3
    X_input = []
    Y_output = []
    for i in range(0, len(text) - Tx, step):
        X_input.append(text[i: i + Tx])
        Y_output.append(text[i + Tx])

    return X_input, Y_output
   


# for creating One hot encoded representation of data
def do_input_OHE(X_input, Y_output, Tx, vocab_chars, char_idx):
    # no. of training examples
    m = len(X_input)
    # create the zero vectors of required size
    X = np.zeros((m, Tx, len(vocab_chars)), dtype=np.bool)
    Y = np.zeros((m, len(vocab_chars)), dtype=np.bool)
    
    #  loop over for every sentence 
    for i, sentence in enumerate(X_input):
        # for Tx timesteps 
        for timestep, char in enumerate(sentence):
            X[i, timestep, char_idx[char]] = 1
        
        Y[i, char_idx[Y_output[i]]] = 1

    return X, Y

'''
Article generator
This neural network generates text similar to the one it has been trained.
'''
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
from data_utils import *

Tx = 40
path = r'data/shakespeare.txt'

# returns the highest probability index
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    return np.argmax(preds)
    

# Function invoked at end of each epoch. Prints generated text.
def on_epoch_end(epoch, logs):
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    
    for i in range(400):
        x_sample_input = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_sample_input[0, t, char_indices[char]] = 1.

        preds = model.predict(x_sample_input, verbose=0)
        next_index = sample(preds)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()


# generates article 
def generate_article():
    generated = ''

    user_input = input("Enter starting few words: ")

    # if length is greater than truncate
    if len(user_input) > Tx:
        user_input = user_input[len(user_input) - Tx - 1: len(user_input) - 1].lower()
        sentence = user_input
    elif len(user_input) < Tx:
        # zero pad the sentence to Tx characters.
        sentence = ('{0:0>' + str(Tx) + '}').format(user_input).lower()

    generated += user_input

    sys.stdout.write("\n\nGenerated Article: \n\n")
    sys.stdout.write(user_input)

    for i in range(500):
        x_sample_input = np.zeros((1, Tx, len(chars)))
        for t, char in enumerate(sentence):
            # ignore zero pad character
            if char != '0':
                x_sample_input[0, t, char_indices[char]] = 1.

        preds = model.predict(x_sample_input, verbose=0)[0]
        next_index = sample(preds)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()


chars, text = load_dataset(path)
print('Total no. of chars:', len(chars))

char_indices, indices_char = create_mapping(chars)

maxlen = 40
sentences, next_chars = tokenize_split(text, maxlen)
print('No. of sequences:', len(sentences))

x, y = do_input_OHE(sentences, next_chars, maxlen, chars, char_indices)

# build the model
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
#model.summary()
print('Model built')
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# load saved weights
model.load_weights(r'models/wt_shakespeare.h5')

# generate article
generate_article()

# for training
#print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
#model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
#model.save_weights('wt.h5')
 

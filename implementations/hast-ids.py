# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

from keras.models import Model
from keras.layers import Dense, Input, MaxPooling1D, Conv1D, GlobalMaxPool1D, Activation
from keras.layers import LSTM, Lambda, concatenate
from keras.layers import TimeDistributed
from keras.utils import plot_model
import tensorflow as tf

LSTM_UNITS = 92
MINI_BATCH = 10
TRAIN_STEPS_PER_EPOCH = 12000
VALIDATION_STEPS_PER_EPOCH = 800
DATA_DIR = '/root/data/PreprocessedISCX2012_5class_pkl/'
CHECKPOINTS_DIR = './iscx2012_cnn_rnn_5class_new_checkpoints/'
TRAIN_EPOCHS = 8
PACKET_NUM_PER_SESSION = 111 # ToDo: This is guessed! came via arg[1] - check real value in paper!
PACKET_LEN = 222 # ToDo: This is guessed! came via arg[0] - check real value in paper!


dict_5class = {0:'Normal', 1:'BFSSH', 2:'Infilt', 3:'HttpDoS', 4:'DDoS'}

def binarize(x, sz=256):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 256

def byte_block(in_layer, nb_filter=(64, 100), filter_length=(3, 3), subsample=(2, 1), pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):
        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    block = GlobalMaxPool1D()(block)
    block = Dense(128, activation='relu')(block)
    return block

# create model
session = Input(shape=(PACKET_NUM_PER_SESSION, PACKET_LEN), dtype='int64')
input_packet = Input(shape=(PACKET_LEN,), dtype='int64')
embedded = Lambda(binarize, output_shape=binarize_outshape)(input_packet)
block2 = byte_block(embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
block3 = byte_block(embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))
packet_encode = concatenate([block2, block3], axis=-1)
encoder = Model(inputs=input_packet, outputs=packet_encode)
encoder.summary()
plot_model(encoder, to_file='hast-ids-encoder.png', show_layer_names=True, show_shapes=True)

encoded = TimeDistributed(encoder)(session)
lstm_layer = LSTM(LSTM_UNITS, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
lstm_layer2 = LSTM(LSTM_UNITS, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0)(lstm_layer)
dense_layer = Dense(5, name='dense_layer')(lstm_layer2)
output = Activation('softmax')(dense_layer)
model = Model(outputs=output, inputs=session)
model.summary()
plot_model(model, to_file='hast-ids-model.png', show_layer_names=True, show_shapes=True)

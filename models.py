from turtle import shape
from typing import final
from xml.sax.xmlreader import InputSource
from cv2 import DenseOpticalFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTM, LSTMCell, Dense, Conv1D, Conv1DTranspose, Concatenate, Cropping1D
import tensorflow_addons as tfa
import numpy as np

def EncoderDecoderBasic(input_shape, output_shape):
    encoder_inputs = Input(shape = input_shape, dtype = np.float32)
    decoder_inputs = Input(shape = output_shape, dtype = np.float32)
    sampler = tfa.seq2seq.TrainingSampler()
    encoder = LSTM(128, return_state=True)
    decoder_cell = LSTMCell(128)
    output_layer = Dense(1, activation = 'tanh')
    decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler, output_layer = output_layer)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_state = [state_h, state_c]
    final_output, final_state, _ = decoder(decoder_inputs, initial_state = encoder_state) 
    model = keras.Model(inputs = [encoder_inputs, decoder_inputs], outputs = [final_output.rnn_output])
    return model

def SimpleLSTM(input_shape, output_shape):
    input = Input(shape = input_shape, dtype = np.float32)
    lstm = LSTM(256)(input)
    output = Dense(11000, activation = 'tanh')(lstm)
    model = keras.Model(inputs = [input], outputs = [output])
    return model

def Unet1D():
    inp = Input(shape=(5500,1))
    c1 = Conv1D(2,32,2,'same',activation='relu')(inp)
    c2 = Conv1D(4,32,2,'same',activation='relu')(c1)
    c3 = Conv1D(8,32,2,'same',activation='relu')(c2)
    c4 = Conv1D(16,32,2,'same',activation='relu')(c3)
    c5 = Conv1D(32,32,2,'same',activation='relu')(c4)

    dc1 = Conv1DTranspose(32,32,1,padding='same')(c5)
    conc = Concatenate()([c5,dc1])
    dc2 = Conv1DTranspose(16,32,2,padding='same')(conc)
    conc = Concatenate()([c4,dc2])
    dc3 = Conv1DTranspose(8,32,2,padding='same')(conc)
    conc = Concatenate()([c3,dc3])
    dc4 = Cropping1D((0,1))(Conv1DTranspose(4,32,2,padding='same')(conc))
    conc = Concatenate()([c2,dc4])
    dc5 = Conv1DTranspose(2,32,2,padding='same')(conc)
    conc = Concatenate()([c1,dc5])
    dc6 = Conv1DTranspose(1,32,2,padding='same')(conc)
    conc = Concatenate()([inp,dc6])
    dc7 = Conv1DTranspose(1,32,1,padding='same',activation='linear')(conc)
    output = Conv1DTranspose(1, 1, 2, padding = 'same', activation = 'tanh')(dc7)
    model = tf.keras.models.Model(inp,output)
    return model




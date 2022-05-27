from cv2 import DenseOpticalFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTM, LSTMCell, Dense
import tensorflow_addons as tfa
import numpy as np

def EncoderDecoderBasic(input_shape, output_shape):
    encoder_inputs = Input(shape = input_shape, dtype = np.float32)
    decoder_inputs = Input(shape = output_shape, dtype = np.float32)
    sampler = tfa.seq2seq.TrainingSampler()
    encoder = LSTM(512, return_state=True)
    decoder_cell = LSTMCell(512)
    decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_state = [state_h, state_c]
    final_output, final_state, _ = decoder(decoder_inputs, initial_state = encoder_state) 
    model = keras.Model(inputs = [encoder_inputs, decoder_inputs], outputs = [final_output])
    return model


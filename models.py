import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTM, LSTMCell, Dense, Conv1D, Conv1DTranspose, Concatenate, Cropping1D, LayerNormalization
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
    norm = LayerNormalization(axis = -2)(inp)
    c1 = Conv1D(2,32,2,'same',activation='relu')(norm)
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
    output = Conv1DTranspose(1, 64, 2, padding = 'same', activation = 'tanh')(dc7)
    model = tf.keras.models.Model(inp,output)
    return model

def Unet1D_v1():
    inp = input(shape=(5500,1))
    norm = LayerNormalization(axis = -2)(inp)
    c1 = Conv1D(16,32,2,'same',activation='relu')(norm)
    c2 = Conv1D(32,32,2,'same',activation='relu')(c1)
    c3 = Conv1D(64,32,2,'same',activation='relu')(c2)
    c4 = Conv1D(128,32,2,'same',activation='relu')(c3)
    c5 = Conv1D(256,32,2,'same',activation='relu')(c4)

    dc1 = Conv1DTranspose(256,32,1,padding='same')(c5)
    conc = Concatenate()([c5,dc1])
    dc2 = Conv1DTranspose(128,32,2,padding='same')(conc)
    conc = Concatenate()([c4,dc2])
    dc3 = Conv1DTranspose(64,32,2,padding='same')(conc)
    conc = Concatenate()([c3,dc3])
    dc4 = Cropping1D((0,1))(Conv1DTranspose(32,32,2,padding='same')(conc))
    conc = Concatenate()([c2,dc4])
    dc5 = Conv1DTranspose(16,32,2,padding='same')(conc)
    conc = Concatenate()([c1,dc5])
    dc6 = Conv1DTranspose(1,32,2,padding='same')(conc)
    conc = Concatenate()([inp,dc6])
    dc7 = Conv1DTranspose(1,32,1,padding='same',activation='linear')(conc)
    output = Conv1DTranspose(1, 64, 2, padding = 'same', activation = 'tanh')(dc7)
    model = keras.models.Model(inp,output)
    return model


def Unet1D_v2():
    inp = Input(shape=(5500,1))
    norm = LayerNormalization(axis = -2)(inp)
    c1 = Conv1D(32,32,2,'same',activation='relu')(norm)
    c2 = Conv1D(64,32,2,'same',activation='relu')(c1)
    c3 = Conv1D(128,32,2,'same',activation='relu')(c2)
    c4 = Conv1D(256,32,2,'same',activation='relu')(c3)
    c5 = Conv1D(512,32,2,'same',activation='relu')(c4)

    dc1 = Conv1DTranspose(512,32,1,padding='same')(c5)
    conc = Concatenate()([c5,dc1])
    dc2 = Conv1DTranspose(256,32,2,padding='same')(conc)
    conc = Concatenate()([c4,dc2])
    dc3 = Conv1DTranspose(128,32,2,padding='same')(conc)
    conc = Concatenate()([c3,dc3])
    dc4 = Cropping1D((0,1))(Conv1DTranspose(64,32,2,padding='same')(conc))
    conc = Concatenate()([c2,dc4])
    dc5 = Conv1DTranspose(32,32,2,padding='same')(conc)
    conc = Concatenate()([c1,dc5])
    dc6 = Conv1DTranspose(1,32,2,padding='same')(conc)
    conc = Concatenate()([inp,dc6])
    dc7 = Conv1DTranspose(1,32,1,padding='same',activation='linear')(conc)
    output = Conv1DTranspose(1, 64, 2, padding = 'same', activation = 'tanh')(dc7)
    model = keras.models.Model(inp,output)
    return model

def Unet1D_v3():
    inp = Input(shape=(5500,1))
    norm = LayerNormalization(axis = -2)(inp)
    
    c1_layer0 = Conv1D(32,32,1,'same',activation='relu')(norm)

    c1_layer1 = Conv1D(32,32,2,'same',activation='relu')(c1_layer0)
    c2_layer1 = Conv1D(32,32,1,'same',activation='relu')(c1_layer1)

    c1_layer2 = Conv1D(64,32,2,'same',activation='relu')(c2_layer1)
    c2_layer2 = Conv1D(64,32,1,'same',activation='relu')(c1_layer2)
    
    c1_layer3 = Conv1D(128,32,2,'same',activation='relu')(c2_layer2)
    c2_layer3 = Conv1D(128,32,1,'same',activation='relu')(c1_layer3)
    
    c1_layer4 = Conv1D(256,32,2,'same',activation='relu')(c2_layer3)
    c2_layer4 = Conv1D(256,32,1,'same',activation='relu')(c1_layer4)
    
    c5 = Conv1D(512,32,2,'same',activation='relu')(c2_layer4)

    dc1_layer5 = Conv1DTranspose(512,32,1,padding='same')(c5)
    dc2_layer5 = Conv1DTranspose(512,32,1,padding='same')(dc1_layer5)
    
    #conc = Concatenate()([c5,dc2_layer5])

    dc1_layer4 = Conv1DTranspose(256,32,2,padding='same')(dc2_layer5)
    conc = Concatenate()([c2_layer4, dc1_layer4])
    dc2_layer4 = Conv1DTranspose(256,32,1,padding='same')(conc)

    dc1_layer3 = Conv1DTranspose(128,32,2,padding='same')(dc2_layer4)
    conc = Concatenate()([c2_layer3, dc1_layer3])
    dc2_layer3 = Conv1DTranspose(128,32,1,padding='same')(conc)
    
    dc1_layer2 = Cropping1D((0,1))(Conv1DTranspose(64,32,2,padding='same')(dc2_layer3))
    conc = Concatenate()([c2_layer2, dc1_layer2])
    dc2_layer2 = Conv1DTranspose(64,32,1,padding='same')(conc)
    
    dc1_layer1 = Conv1DTranspose(32,32,2,padding='same')(dc2_layer2)
    conc = Concatenate()([c2_layer1, dc1_layer1])
    dc2_layer1 = Conv1DTranspose(32,32,1,padding='same')(conc)
    
    dc1_layer0 = Conv1DTranspose(32,32,2,padding='same')(dc2_layer1)
    conc = Concatenate()([c1_layer0, dc1_layer0])
    dc2_layer0 = Conv1DTranspose(32,32,1,padding='same')(conc)
    
    dc7 = Conv1DTranspose(1,32,1,padding='same',activation='linear')(dc2_layer0)
    output = Conv1DTranspose(1, 64, 2, padding = 'same', activation = 'tanh')(dc7)
    model = keras.models.Model(inp,output)
    return model

def Unet1D_v4():
    inp = Input(shape=(5500,1))
    norm = LayerNormalization(axis = -2)(inp)
    c1 = Conv1D(16,32,2,'same',activation='relu')(norm)
    c2 = Conv1D(32,32,2,'same',activation='relu')(c1)
    c3 = Conv1D(64,32,2,'same',activation='relu')(c2)
    c4 = Conv1D(128,32,2,'same',activation='relu')(c3)
    c5 = Conv1D(256,32,2,'same',activation='relu')(c4)

    dc1 = Conv1DTranspose(256,32,1,padding='same')(c5)
    dc2 = Conv1DTranspose(128,32,2,padding='same')(dc1)
    dc3 = Conv1DTranspose(64,32,2,padding='same')(dc2)
    dc4 = Conv1DTranspose(32,32,2,padding='same')(dc3)
    dc4_cr = Cropping1D((0,1))(dc4)
    dc5 = Conv1DTranspose(16,32,2,padding='same')(dc4_cr)
    dc6 = Conv1DTranspose(1,32,2,padding='same')(dc5)
    dc7 = Conv1DTranspose(1,32,1,padding='same',activation='linear')(dc6)
    output = Conv1DTranspose(1, 64, 2, padding = 'same', activation = 'tanh')(dc7)
    model = keras.models.Model(inp,output)
    return model


# -*- encoding: utf-8 -*-

from keras.layers import Input, Embedding, LSTM, Convolution1D
from keras.layers import MaxPooling1D, AveragePooling1D, Bidirectional
from keras.layers import merge, Dropout, Flatten, Dense, Permute
from keras.models import Model, Sequential
from keras.regularizers import l2

def buildLSTM(*args):
	main_inputs = Input(shape=(args[0],), dtype='int32', name='main_input')
	inputs = Embedding(input_dim=args[1],
					   output_dim=100,
					   weights=[args[2]],
					   input_length=args[0])(main_inputs)
	x = LSTM(100)(inputs)
	x = Dense(50, activation='relu')(x)
	predict = Dense(2, activation='softmax')(x)
	model = Model(input=main_inputs, output=predict)
	return model


def buildCNN(*args):
	main_inputs = Input(shape=(args[0],), dtype='int32', name='main_input')
	inputs = Embedding(input_dim=args[1],
					   output_dim=100,
					   weights=[args[2]],
					   input_length=args[0])(main_inputs)
	cnns = [Convolution1D(filter_length=filter_length,
						  nb_filter=args[3],
						  # W_regularizer=l2(0.01),
						  activation='relu',
						  border_mode='same') for filter_length in [2, 3, 5]]

	max_1 = MaxPooling1D(args[0])
	feature = [max_1(cnn(inputs)) for cnn in cnns]
	x = merge(feature, mode='concat', concat_axis=2)  # (None, 1, 100)

	x = Dropout(0.25)(x)
	x = Flatten()(x)
	x = Dense(150, activation='relu')(x)
	x = Dense(50, activation='relu')(x)
	predict = Dense(2, activation='softmax')(x)
	model = Model(input=main_inputs, output=predict)
	return model


def build_LSTM_CNN(*args):
	input = Input(shape=(args[0],), dtype='int32')
	embedding = Embedding(input_dim=args[1],
						  output_dim=100,
						  weights=[args[2]],
						  # trainable=False,
						  input_length=args[0])(input)
	x = LSTM(100, return_sequences=True, W_regularizer=l2(0.01), dropout_W=0.2)(embedding)
	# x = GRU(100, return_sequences=True, W_regularizer=l2(0.01), dropout_W=0.2, name='gru1')(embedding)
	cnns = [Convolution1D(filter_length=filter_length,
						  nb_filter=args[3],
						  # W_regularizer=l2(0.01),
						  activation='relu',
						  border_mode='same') for filter_length in [2, 3, 5]]  # [2, 3, 5, 7]

	max_1 = MaxPooling1D(args[0])
	feature = [max_1(cnn(x)) for cnn in cnns]
	x = merge(feature, mode='concat', concat_axis=2)  # (None, 1, 100)

	# x = merge([GlobalMaxPooling1D()(cnn(x)) for cnn in cnns], mode='concat', concat_axis=1)
	x = Dropout(0.25)(x)
	x = Flatten()(x)
	# x = Dense(300, activation='relu')(x)
	x = Dense(150, activation='relu')(x)
	x = Dense(50, activation='relu')(x)
	output = Dense(2, activation='softmax')(x)
	model = Model(input=[input], output=[output])
	return model


def build_CNN_LSTM(*args):
	input = Input(shape=(args[0],), dtype='int32')
	embedding = Embedding(input_dim=args[1],
						  output_dim=100,
						  weights=[args[2]],
						  # trainable=False,
						  input_length=args[0])(input)
	# embedding = Dropout(0.25)(embedding)
	cnns = [Convolution1D(filter_length=filter_length,
						  nb_filter=args[3],
						  activation='relu',
						  border_mode='same') for filter_length in [2, 3, 5]]

	max_1 = MaxPooling1D(pool_size=args[0])
	feature = [max_1(cnn(embedding)) for cnn in cnns]
	x = merge(feature, mode='concat', concat_axis=2)  # (None, 1, 100)
	# x = merge([GlobalMaxPooling1D()(cnn(x)) for cnn in cnns], mode='concat', concat_axis=1)

	x = LSTM(300, W_regularizer=l2(0.01), dropout_W=0.2)(x)
	x = Dense(150, activation='relu')(x)
	x = Dense(50, activation='relu')(x)
	output = Dense(2, activation='softmax')(x)
	model = Model(input=[input], output=[output])
	return model


def merge_LSTM_CNN(*args):
	'''横向, CNN 窗口为3 '''
	input = Input(shape=(args[0],), dtype='int32')
	embedding = Embedding(input_dim=args[1],
						  output_dim=100,
						  weights=[args[2]],
						  # trainable=False,
						  input_length=args[0])(input)
	x = LSTM(100, return_sequences=True)(embedding)

	cnn = Convolution1D(filter_length=3, nb_filter=args[3],
						  activation='relu', border_mode='same')(embedding)

	x = merge([x, cnn], mode='concat', concat_axis=2)
	# out = MaxPooling1D(args[0])(cnn)

	x = Flatten()(x)
	x = Dense(100, activation='relu')(x)
	x = Dense(50, activation='relu')(x)
	output = Dense(2, activation='softmax')(x)
	model = Model(input=[input], output=[output])
	return model
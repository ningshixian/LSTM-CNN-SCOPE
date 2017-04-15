# -*- encoding: utf-8 -*-
from __future__ import print_function
import numpy as np

import cPickle  # import pickle as cPickle
import model

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import acc_loss
import PRF
from process_data import make_idx_data_cv
import time

np.random.seed(1337)

MAX_NB_WORDS = 22353   # 字典大小
MAX_SEQUENCE_LENGTH = 10  # 每个句子最多保留10个词
EMBEDDING_DIM = 100  # 词向量的维度

model_path = r'weight/LSTM_maxpool_lweights.h5'
RESULT_FILE = r'BiLSTM_L_predict_result.txt'
DATA_PATH = 'data/mr_Lscope.p'

# Training
VALIDATION_SPLIT = 0.2  # 训练集:验证集 = 1:4
NB_EPOCHS = 20  # 迭代次数
BATCH_SIZE = 64

# Convolution
FILTER_LENGTH = [2, 3, 5]
NB_FILTER = 100
POOL_LENGTH = 8


def loadData(path):
	x = cPickle.load(open(path, "rb"))
	revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
	print(len(word_idx_map))
	print(len(vocab))
	datasets = make_idx_data_cv(revs, word_idx_map, 1, max_l=10, k=100, filter_h=1)
	img_h = len(datasets[0][0]) - 1
	test_set_x = datasets[1][:, :img_h]
	test_set_y = np.asarray(datasets[1][:, -1], "int32")
	train_set_x = datasets[0][:, :img_h]
	train_set_y = np.asarray(datasets[0][:, -1], "int32")
	print(np.shape(train_set_x))
	print('load data...')
	print(np.shape(W))
	print(type(W))
	return (train_set_x, train_set_y), (test_set_x, test_set_y), W


if __name__ == '__main__':
	################################################## Load data ###########################################################
	print('\n-------Start loading data---------')

	(X_train, y_train), (X_test, y_test), word_embedding_matrix = loadData(DATA_PATH)
	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')

	nb_words = min(MAX_NB_WORDS, len(word_embedding_matrix))

	#   5-Prepare training data tensors
	print('Pad sequences (samples x time)')
	X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
	X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)
	train_label = to_categorical(y_train, 2)
	test_label = to_categorical(y_test, 2)
	print('Build model...')

	##############################################  Model prepare  #########################################################
	model = model.build_LSTM_CNN(MAX_SEQUENCE_LENGTH, nb_words,
						   word_embedding_matrix, NB_FILTER)
	model = model.build_CNN_LSTM(MAX_SEQUENCE_LENGTH, nb_words,
						   word_embedding_matrix, NB_FILTER)

	model.compile(loss='categorical_crossentropy',
				  optimizer='adagrad',  # rmsprop
				  metrics=['accuracy'])
	model.summary()  # 打印出模型概况
	#############################################  Train Model  ############################################################

	# 该回调函数将在每个epoch后保存模型到 filepath
	checkpoint = ModelCheckpoint(filepath=best_model_tmp, monitor='val_loss',
					verbose=1, save_best_only=True, mode='min')

	t0 = time.time()
	history = model.fit(X_train, train_label,
						batch_size=BATCH_SIZE,
						validation_data=(X_test, test_label),
						# validation_split=VALIDATION_SPLIT,
						callbacks=[checkpoint],
						nb_epoch=NB_EPOCHS)
	t1 = time.time()
	print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

	# 将模型和权重保存到指定路径
	# model.save(model_path)
	# 加载权重到当前模型
	# model = load_model(model_path)

	# Print best validation accuracy and epoch in valid_set
	max_val_loss, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
	print('Maximum accuracy at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(max_val_loss))

	# plot the result
	acc_loss.plot(history)

	score, acc = model.evaluate(X_test, test_label, batch_size=BATCH_SIZE)
	print('Test score:', score)
	print('Test accuracy:', acc)

	predictions = model.predict(X_test)
	PRF.calculate(predictions, test_label, RESULT_FILE)

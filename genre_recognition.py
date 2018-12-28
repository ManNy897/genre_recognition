import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	def on_epoch_end(self, epoch, logs={}):
		softmax_output = np.asarray(self.model.predict(self.validation_data[0]))
		val_predict = np.argmax(softmax_output, axis=1)
		val_targ = self.validation_data[1]
		_val_f1 = f1_score(val_targ, val_predict, average='macro', labels=np.unique(val_predict))
		_val_recall = recall_score(val_targ, val_predict, average='macro', labels=np.unique(val_predict))
		_val_precision = precision_score(val_targ, val_predict, average='macro', labels=np.unique(val_predict))
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print(' — val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
		return
	def get_data(self):
		return self.val_f1s

def load_data(feats_path, labels_path):
	print('loading data...')
	feats = pd.read_csv(feats_path)
	labels = pd.read_csv(labels_path, skiprows=1)
	feats = feats[3:] #remove unused rows
	feats = feats.drop(feats.columns[[0]], axis=1) #drop first unused column
	feats = feats.astype(float)
	feats = feats.reset_index(drop=True) #make feats and bool_mask have sae indexes
	labels = labels['genre_top']
	labels = labels[1:]
	labels = labels.reset_index(drop=True)
	bool_mask = labels.notnull()
	feats = feats[bool_mask] #only choose feats and labels that have top level genre. (some of the rows have this data missing)
	labels = labels[bool_mask]
	categories = labels.unique()
	labels = labels.astype("category", categories=categories).cat.codes
	print('finished loading data')
	return feats, labels, categories

def pre_process(feats, num_feats):
	print('pre processsing data...')
	norm_feats = (feats-feats.mean())/(feats.max()-feats.min())
	eig_vals, eig_vecs = np.linalg.eig(np.dot(norm_feats.T, norm_feats))
	# print('sum of eigen values: ' + str(np.sum(eig_vals)))
	# print('90 percent of eigen values: ' + str(np.sum(eig_vals) * 0.9))
	# print('sum of first 80 eigen values: ' + str(np.sum(eig_vals[:80])))
	pca_feats = np.dot(norm_feats, eig_vecs[:,:num_feats]) #only choose first num_feats pca features
	print('finished pre processing data')
	return pca_feats

def build_model(num_layers, input_dim, metric):
	model = keras.Sequential()
	model.add(keras.layers.Dense(1024, activation=tf.nn.relu, input_shape=(input_dim,)))
	for _ in range(num_layers):
		model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
	model.add(keras.layers.Dense(16, activation=tf.nn.softmax))
	model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
	return model

def build_data(feats, labels, num_feats):
	feats = pre_process(feats, num_feats)
	feats_train, feats_test, labels_train, labels_test = train_test_split(feats, labels, test_size=0.20)
	feats_val, feats_test, labels_val, labels_test = train_test_split(feats_test, labels_test, test_size=0.50)
	return feats_train, feats_val, feats_test, labels_train, labels_val, labels_test

def train(feats, labels, feats_val, labels_val, model, metric):
	print('training model...')
	# input_dim = feats.shape[1]
	#print('training model with ' + str(num_layers) + ' layers')
	##build model with num_layers as the number of layers
	# model = keras.Sequential()
	# model.add(keras.layers.Dense(1024, activation=tf.nn.relu, input_shape=(input_dim,)))
	# for _ in range(num_layers):
	# 	model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
	# model.add(keras.layers.Dense(16, activation=tf.nn.softmax))
	# model = keras.Sequential([
	# 	keras.layers.Dense(1024, activation=tf.nn.relu, input_shape=(input_dim,)),
	# 	keras.layers.Dense(1024, activation=tf.nn.relu),
	# 	keras.layers.Dense(16, activation=tf.nn.softmax)
	# ])
	# model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy', metric])
	model.fit(feats, labels, epochs=1, batch_size=32, validation_data=(feats_val, labels_val), callbacks=[metric])
	print('finished training model')
	return model

# def validation(feats_train, labels_train, feats_val, labels_val, metric):
def validation(feats, labels, metric):
	print('running validation of different model archictures')
	layer_nums = [1,5,10,20]
	# layer_nums = [1]
	num_feats = [10, 80, 300, 518]
	# num_feats = [300]
	#make models for all of the layer number of layers
	models = []
	scores = []
	for j in num_feats:
		feats_train, feats_val, feats_test, labels_train, labels_val, labels_test = build_data(feats, labels, j)
		input_dim = feats_train.shape[1]
		for i in layer_nums:
			print('training model with ' + str(i) + ' layers and ' + str(j) + ' feats')
			m = build_model(i, input_dim, metric)
			trained_model = train(feats_train, labels_train, feats_val, labels_val, m, metric)
			models.append(trained_model)
			scores.append(test(feats_val, labels_val, trained_model))
	#models = [train(feats_train, labels_train, feats_val, labels_val, i, metric) for i in layer_nums]
	# scores = [test(feats_val, labels_val, model) for model in models]
	scores = np.array(scores)
	max_index = np.argmax(scores[:,1]) #take indicie of model with max accuracy
	#using the best model, regenerate the corresponding data according to the best model hyperparams
	max_feat_num_index = int((max_index / (float(len(num_feats) * len(layer_nums)))) / (1.0/len(num_feats)))
	print(max_feat_num_index)
	max_feat_num = num_feats[max_feat_num_index]
	feats_train, feats_val, feats_test, labels_train, labels_val, labels_test = build_data(feats, labels, max_feat_num)
	best_model = models[max_index]
	best_model_train_score = test(feats_train, labels_train, best_model)
	print('scores for val models: ')
	for j in range(len(num_feats)):
		for i in range(len(layer_nums)):
			print(str(layer_nums[i]) + ' layers and ' + str(num_feats[j]) + ' : ' + str(scores[j*len(layer_nums)+i]))
	print('best model is model: ' + str(max_index))
	print('best model train score: ' + str(scores_to_dict(best_model_train_score)))
	print('best model val score: ' + str(scores_to_dict(scores[max_index])))
	print('finished validation step')
	return best_model, feats_test, labels_test

def scores_to_dict(scores):
	scores_labels = ['loss', 'acc', 'recall', 'prec', 'f1']
	scores_dict = dict((scores_labels[i], scores[i]) for i in range(len(scores)))
	return scores_dict


def test(feats, labels, model):
	print('running evaluation...')
	score = model.evaluate(feats, labels, verbose=0)
	softmax_output = np.asarray(model.predict(feats))
	test_predict = np.argmax(softmax_output, axis=1)
	test_targ = labels
	_test_f1 = f1_score(test_targ, test_predict, average='macro', labels=np.unique(test_predict))
	_test_recall = recall_score(test_targ, test_predict, average='macro', labels=np.unique(test_predict))
	_test_precision = precision_score(test_targ, test_predict, average='macro', labels=np.unique(test_predict))
	all_scores = np.array([score[0], score[1], _test_recall, _test_precision, _test_f1])
	print('finished evaluation...')
	return all_scores



def main():
	feats_dir = "fma_metadata/features.csv"
	labels_dir = "fma_metadata/tracks.csv"
	# num_feats = 300
	feats, labels, categories = load_data(feats_dir,labels_dir)
	# feats = pre_process(feats, num_feats)
	# feats_train, feats_test, labels_train, labels_test = train_test_split(feats, labels, test_size=0.20)
	# feats_val, feats_test, labels_val, labels_test = train_test_split(feats_test, labels_test, test_size=0.50)
	#model = train(feats_train, labels_train)
	metric = Metrics()
	# model = validation(feats_train, labels_train, feats_val, labels_val, metric)
	model, feats_test, labels_test = validation(feats, labels, metric)
	score = test(feats_test, labels_test, model)
	scores_dict = scores_to_dict(score)
	
	print('testing score: ' + str(scores_dict))


if __name__ == "__main__":
	main()


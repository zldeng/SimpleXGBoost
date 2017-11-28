#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2017-11-28 16:35:07
'''
 
import sys
sys.path.append('../src/')

from gboost import sgboost
import pandas as pd
import numpy as np


def generateData(data_cnt):
	np.random.seed(12)
	x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], data_cnt)
	x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], data_cnt)
	
	simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
	simulated_labels = np.hstack((np.zeros(data_cnt),np.ones(data_cnt)))
	
	#print type(simulated_separableish_features)
	#print type(simulated_labels)
	
	X_datafram = pd.DataFrame(simulated_separableish_features)
	y_series = pd.Series(simulated_labels)

	return X_datafram,y_series



train_data_cnt = 1000
train_X,train_y = generateData(train_data_cnt)


val_data_cnt = 200
val_X,val_y = generateData(val_data_cnt)

test_data_cnt = 300
test_X,test_y = generateData(test_data_cnt)


#print train_X
#print '\n'
#print train_y

params = {'loss': "logisticloss",
	'eta': 0.3,
	'max_depth': 6,
	'num_boost_round': 10,
	'scale_pos_weight': 1.0,
	'subsample': 0.7,
	'colsample_bytree': 0.7,
	'colsample_bylevel': 1.0,
	'min_sample_split': 10,
	'min_child_weight': 2,
	'reg_lambda': 10,
	'gamma': 0,
	'eval_metric': "error",
	'early_stopping_rounds': 20,
	'maximize': False,
	'num_thread': 16}

tgb = sgboost()
tgb.fit(train_X, train_y, validation_data=(val_X, val_y), **params)

pred_res = tgb.predict(test_X)


correct_cnt = np.sum((pred_res == test_y).astype(int))
test_cnt = len(test_y)

acc = correct_cnt * 1.0 / test_cnt

print test_cnt,correct_cnt,acc

#test save model
model_name = 'tmp.model'
tgb.saveModel(model_name)

#test load model and predict
tmp_tgb = sgboost()
tmp_tgb.loadModel(model_name)

tmp_pred = tmp_tgb.predict(test_X)

corr_cnt = np.sum((pred_res == test_y).astype(int))

print corr_cnt,corr_cnt*1.0/test_cnt


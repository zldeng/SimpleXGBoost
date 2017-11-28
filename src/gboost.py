#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2017-11-24 16:05:41
'''
import sys 
from collections import defaultdict
import pandas as pd
import numpy as np
from lossFunc import LogisticLoss
from gboostTree import Tree
from metric import get_metric
import pickle

class sgboost(object):
	'''
	simple xgboost	
	'''
	def __init__(self):
		self.trees = []
		self.eta = None
		self.num_boost_round = None
		self.first_round_pred = None
		self.loss = None
		self.subsample = None
		self.max_depth = None
		self.colsample_bylevel = None
		self.colsample_bytree = None
		self.reg_lambda = None
		self.min_sample_split = None
		self.gamma = None
		self.num_thread = None
		self.min_child_weight = None
		self.scale_pos_weight = None
		self.feature_importance = defaultdict(lambda:0)
		self.pred_cutoff = 0.5

		self.attr_need_save = ['eta','first_round_pred','reg_lambda','gamma','pred_cutoff','loss']

	def fit(self,X,y,validation_data = (None,None),early_stopping_rounds = 10,
			maximize = True,eval_metric = None,loss = 'logisticloss',eta = 0.3,
			num_boost_round = 1000,max_depth = 5,scale_pos_weight = 1,
			subsample = 0.8,colsample_bytree = 0.8,colsample_bylevel = 0.8,
			min_child_weight = 1,min_sample_split = 10,reg_lambda = 1.0,gamma = 0,
			num_thread = -1,pred_cutoff = 0.5):
		'''
		X:pandas.core.frame.DataFrame
		y:pandas.core.series.Series
		early_stopping_rounds: early_stop when eval rsult become worse more the early_stopping_rounds times
		maximize:the target is to make loss as large as possible
		eval_metric: evaluate method
		loss : loss function for optionmize
		num_boost_round : number of boosting
		max_depth: max_depth for a tree
		scale_pos_weight: weight for samples with 1 labels
		subsample: row sample rate when build a tree
		colsample_bytree: column sample rate when building a tree
		colsample_bylevel: column sample rate when spliting each tree node. when split a tree,the number of features = total_features*colsample_bytree*colsample_bylevel
		min_sample_split: min number of samples in a leaf node
		'''
		self.eval_metric = eval_metric
		self.eta = eta
		self.num_boost_round = num_boost_round
		self.first_round_pred  = 0.0
		self.subsample = subsample
		self.max_depth = max_depth
		self.colsample_bytree = colsample_bytree
		self.colsample_bylevel = colsample_bylevel
		self.reg_lambda = reg_lambda
		self.min_sample_split = min_sample_split
		self.gamma = gamma
		self.num_thread = num_thread
		self.min_child_weight = min_child_weight
		self.scale_pos_weight = scale_pos_weight
		self.pred_cutoff = pred_cutoff
		
		#将X,y修改为能通过int下标（从0开始）进行索引的FramData
		X.reset_index(drop = True,inplace = True)
		y.reset_index(drop = True,inplace = True)

		if 'logisticloss':
			self.loss = LogisticLoss(self.reg_lambda)
		elif 'squareloss' == loss:
			self.loss = SquareLoss(self.reg_lambda)
		else:
			raise Exception('No find match loss')

		if not isinstance(validation_data,tuple):
			raise Exception('validation_data must be tuple')
		
		val_X,val_y = validation_data

		do_val = True
		if val_X is None or val_y is None:
			do_val = False
		else:
			if not isinstance(val_X,pd.core.frame.DataFrame):
				raise Exception('val_X must be pd.core.frame.DataFrame')			
			
			if not isinstance(val_y,pd.core.series.Series):
				raise Exception('val_y must be pd.core.series.Series')

			val_X.reset_index(drop = True,inplace = True)
			val_y.reset_index(drop = True,inplace = True)

			val_Y = pd.DataFrame(val_y.values,columns = ['label'])
			
			#set default pred value
			val_Y['y_pred'] = self.first_round_pred

		if maximize:
			best_val_metric = -np.inf
			best_round = 0
			become_worse_round = 0
		else:
			best_val_metric = np.inf
			best_round = 0
			become_worse_round = 0

		Y = pd.DataFrame(y.values,columns = ['label'])
		Y['y_pred'] = self.first_round_pred
		Y['grad'] = self.loss.grad(Y.y_pred.values,Y.label.values)
		Y['hess'] = self.loss.hess(Y.y_pred.values,Y.label.values)

		Y['sample_weight'] = 1.0
		#调整正样本权重
		Y.loc[Y.label == 1,'sample_weight'] = self.scale_pos_weight
		
		for i in range(self.num_boost_round):
			# row and column sample before training the current tree
			data = X.sample(frac=self.colsample_bytree, axis=1) #column sample
			data = pd.concat([data, Y], axis=1)
			data = data.sample(frac=self.subsample, axis=0) #row sample

			Y_selected = data[['label', 'y_pred', 'grad', 'hess']]
			X_selected = data.drop(['label', 'y_pred', 'grad', 'hess', 'sample_weight'], axis=1)
			
			#print X_selected
			#print Y_selected

			# fit a tree
			tree = Tree()
			tree.fit(X_selected, Y_selected, max_depth=self.max_depth, 
					min_child_weight = self.min_child_weight,
					colsample_bylevel = self.colsample_bylevel,
					min_sample_split = self.min_sample_split,
					reg_lambda = self.reg_lambda,
					gamma = self.gamma,
					num_thread =self.num_thread)

			# predict the whole trainset and update y_pred,grad,hess
			preds = tree.predict(X)
			Y['y_pred'] += self.eta * preds
			Y['grad'] = self.loss.grad(Y.y_pred.values, Y.label.values) * Y.sample_weight
			Y['hess'] = self.loss.hess(Y.y_pred.values, Y.label.values) * Y.sample_weight

			# update feature importance
			for k in tree.feature_importance.iterkeys():
				self.feature_importance[k] += tree.feature_importance[k]

			self.trees.append(tree)

			# print training information
			if self.eval_metric is None or not do_val:
				print "GBoost round {iteration}".format(iteration=i)
			
			#evaluate in validation data
			else:
				try:
					mertric_func = get_metric(self.eval_metric)
				except:
					raise NotImplementedError("The given eval_metric is not provided")

				train_metric = mertric_func(self.loss.transform(Y.y_pred.values), Y.label.values)

				#val_Y is [n_sampels 2], column is label,pred
				val_Y['y_pred'] += self.eta * tree.predict(val_X)

				#evaludate on the current predict result
				val_metric = mertric_func(self.loss.transform(val_Y.y_pred.values), val_Y.label.values)

				print "GBoost round {iteration}, train-{eval_metric} is {train_metric}, val-{eval_metric} is {val_metric}".format(
					iteration=i, eval_metric=self.eval_metric, train_metric=train_metric, val_metric=val_metric)

				# check if to early stop
				if maximize:
					if val_metric > best_val_metric:
						best_val_metric = val_metric
						best_round = i
						become_worse_round = 0
					else:
						become_worse_round += 1
					
					#when the evaluation result is worse more than early_stopping_rounds times
					#stop to continue building tree
					if become_worse_round > early_stopping_rounds:
						print "training early Stop, best round is {best_round}, best {eval_metric} is {best_val_metric}".format(
							best_round=best_round, eval_metric=eval_metric, best_val_metric=best_val_metric)
						break
				else:
					if val_metric < best_val_metric:
						best_val_metric = val_metric
						best_round = i
						become_worse_round = 0
					else:
						become_worse_round += 1
					if become_worse_round > early_stopping_rounds:
						print "training early Stop, best round is {best_round}, best val-{eval_metric} is {best_val_metric}".format(
							best_round=best_round, eval_metric=eval_metric, best_val_metric=best_val_metric)
						break


	def predict(self,X):
		assert len(self.trees) > 0

		preds = np.zeros(X.shape[0])
		
		preds += self.first_round_pred
		for tree in self.trees:
			preds += self.eta * tree.predict(X)

		res = self.loss.transform(preds)
		
		return (res > self.pred_cutoff).astype(int)

	def saveModel(self,model_name):
		try:
			obj_dict = self.__dict__
			
			attr_dict = dict((attr,getattr(self,attr)) for attr in dir(self) if attr in self.attr_need_save)
			#print attr_dict
		
			pickle.dump(attr_dict,file(model_name + '.attr','wb'),True)
			pickle.dump(self.trees,file(model_name + '.trees','wb'),True)
			
		except Exception,e:
			print 'Save model file. err=' + str(e)

	def loadModel(self,model_name):
		try:
			attr_dict = pickle.load(file(model_name + '.attr','rb'))
			for attr in attr_dict:
				setattr(self,attr,attr_dict[attr])
			self.trees = pickle.load(file(model_name + '.trees','rb'))

			#print self.__dict__
		except Exception,e:
			print 'Load model fail. err=' + str(e)

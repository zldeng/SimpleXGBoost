#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2017-11-24 10:29:56
'''
import sys 
from multiprocessing import Pool
from functools import partial
import pandas as pd
import numpy as np
import copy_reg
import types
import traceback

def _pickle_method(m):
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)



class TreeNode(object):
	def __init__(self,is_leaf = False,leaf_score = None,split_feature = None,\
			split_threshold = None,left_child = None,
			right_child = None,nan_direction = 0):
		'''
		is_leaf: if True, only need to initialize leaf_score. other params are not used
		leaf_score: prediction score of the leaf node
		split_feature: split feature of the intermediate node
		split_threshold: split threshold of the intermediate node
		left_child: left child node
		right_child: right child node
		nan_direction: if 0, those NAN sample goes to left child, else goes to right child.
		'''
		self.is_leaf = is_leaf
		self.leaf_score = leaf_score
		self.split_feature = split_feature
		self.split_threshold = split_threshold
		self.left_child = left_child
		self.right_child = right_child
		self.nan_direction = nan_direction



class Tree(object):
	def __init__(self):
		self.root = None
		self.min_sample_split = None
		self.colsample_bylevel = None
		self.reg_lambda = None
		self.gamma = None
		self.num_thread = None
		self.min_child_weight = None
		self.feature_importance = {}

	def calculateLeafScore(self,Y):
		'''
		leaf_score = -G/(H+lambda)
		'''

		score = -Y.grad.sum() / (Y.hess.sum() + self.reg_lambda)

		return score

	def calculateSplitGain(self,left_Y,right_Y,G_nan,H_nan,nan_direction = 0):
		'''
		gain = 0.5*(GL^2/(HL+lambda) + GR^2/(HR+lambda) - (GL+GR)^2/(HL+HR+lambda)) - gamma

		G_nan,Hnan is gain from NAN feature value data
		'''
		GL = left_Y.grad.sum() + (1-nan_direction) * G_nan
		HL = left_Y.hess.sum() + (1-nan_direction) * H_nan

		GR = right_Y.grad.sum() + nan_direction * G_nan
		HR = right_Y.grad.sum() + nan_direction * H_nan

		gain = 0.5 * (GL**2 / (HL + self.reg_lambda) + GR**2 / (HR + self.reg_lambda) \
			   - (GL + GR)**2 / (HL + HR + self.reg_lambda)) - self.gamma

		return gain

	def findBestThreshold(self,data,col):
		'''
		find best threshold for the given col feature
		data: the column of data: col,'label','grad','hess''
		'''
		best_threshold = None
		best_gain = -np.inf
		nan_direction = 0
		try:
			selected_data = data[[col,'label','grad','hess']]

			# get the data with/without NAN feature value
			mask = selected_data[col].isnull()
			data_nan = selected_data[mask]
			G_nan = data_nan.grad.sum()
			H_nan = data_nan.hess.sum()
			
			data_not_nan = selected_data[~mask]
			#sort data by the selected feature
			data_not_nan.reset_index(inplace=True)
			data_not_nan.is_copy = False
			data_not_nan[str(col)+'_idx'] = data_not_nan[col].argsort()
			data_not_nan = data_not_nan.ix[data_not_nan[str(col)+'_idx']]

			# linear scan and find the best threshold
			for i in xrange(data_not_nan.shape[0]-1):
				#don't need to split at those same value
				cur_value, nxt_value = data_not_nan[col].iloc[i], data_not_nan[col].iloc[i+1]
				if cur_value == nxt_value:
					continue

				# split at this value
				this_threshold = (cur_value + nxt_value) / 2.0
				this_gain = None
				left_Y = data_not_nan.iloc[:(i+1)]
				right_Y = data_not_nan.iloc[(i+1):]

				# let the NAN data go to left and right, and chose the way which gets the max gain
				nan_goto_left_gain = self.calculateSplitGain(left_Y,right_Y,\
						G_nan,H_nan,nan_direction=0)

				nan_goto_right_gain = self.calculateSplitGain(left_Y, right_Y,\
						G_nan, H_nan, nan_direction=1)

				if nan_goto_left_gain < nan_goto_right_gain:
					cur_nan_direction = 1
					this_gain = nan_goto_right_gain
				else:
					cur_nan_direction = 0
					this_gain = nan_goto_left_gain

				if this_gain > best_gain:
					best_gain = this_gain
					best_threshold = this_threshold
					nan_direction = cur_nan_direction
		except Exception,e:
			traceback.print_exc()
			sys.exit(1)

		return col, best_threshold, best_gain, nan_direction

	def findBestFeatureAndThreshold(self,X,Y):
		"""
		para:
			X [selected_n_samples,selected_feature_samples]
			Y [selected_n_samples,5] column is [label,y_pred,grad,hess,sample_weight]

		find the (feature,threshold) with the largest gain
		if there are NAN in the feature, find its best direction to go
		"""
		nan_direction = 0
		best_gain = - np.inf
		best_feature, best_threshold = None, None
		rsts = None

		# for each feature, find its best_threshold and best_gain
		#finally select the largest gain
		cols = list(X.columns)
		data = pd.concat([X, Y], axis=1)
		
		"""
		print 'findBest:'
		print X
		print Y

		print 'data_info:'
		print data.index
		print data.columns
		print type(data)
		"""

		func = partial(self.findBestThreshold, data)

		if self.num_thread == -1:
			pool = Pool()
			rsts = pool.map(func, cols)
			pool.close()

		else:
			pool = Pool(self.num_thread)
			rsts = pool.map(func, cols)
			pool.close()

		for rst in rsts:
			if rst[2] > best_gain:
				best_gain = rst[2]
				best_threshold = rst[1]
				best_feature = rst[0]
				nan_direction = rst[3]

		return best_feature, best_threshold, best_gain, nan_direction

	def splitData(self, X, Y, feature, threshold, nan_direction):
		"""
			split the dataset according to (feature,threshold), nan_direction
			faeture_value < feature_threshold : left
			faeture_value >= feature_threshold : right
			feature_value==NAN and nan_direction==0 : left
			feature_value==NAN and nan_direction==1 : right
		"""
		X_cols, Y_cols = list(X.columns), list(Y.columns)
		data = pd.concat([X, Y], axis=1)
		right_data = None
		left_data = None
		
		#print 'split:'
		#print data.index
		#print data.columns

		#print feature
		#print threshold

		if nan_direction == 0:
			mask = data[feature] >= threshold
			right_data = data[mask]
			left_data = data[~mask]
		else:
			mask = data[feature] < threshold
			right_data = data[~mask]
			left_data = data[mask]

		return left_data[X_cols], left_data[Y_cols], right_data[X_cols], right_data[Y_cols]

	
	def buildTree(self,X,Y,max_depth):
		'''
		build a tree recursively	
		'''
		if X.shape[0] < self.min_sample_split or max_depth == 0\
			or Y.hess.sum() < self.min_child_weight:
			is_leaf = True
			leaf_score = self.calculateLeafScore(Y)

			leaf_node = TreeNode(is_leaf = is_leaf,leaf_score = leaf_score)
			
			return leaf_node

		#column sample	
		X_selected = X.sample(frac = self.colsample_bylevel,axis = 1)

		best_feature,best_threshold,best_gain,nan_direction = self.findBestFeatureAndThreshold(X_selected,Y)

		if best_gain <= 0:
			is_leaf = True
			leaf_score = self.calculateLeafScore(Y)
			
			leaf_node = TreeNode(is_leaf = is_leaf,leaf_score = leaf_score)
			
			return leaf_node
		
		#split data according to (best_feature,best_threshold,nan_direction)
		left_X,left_Y,right_X,right_Y = self.splitData(X,Y,best_feature,\
				best_threshold,nan_direction)
		
		#creat left tree and right tree
		left_tree = self.buildTree(left_X,left_Y,max_depth-1)
		right_tree = self.buildTree(right_X,right_Y,max_depth-1)

		#update feature importance
		if self.feature_importance.has_key(best_feature):
			self.feature_importance[best_feature] += 1
		else:
			self.feature_importance[best_feature] = 1
		
		#merge left child and right child to get a sub-tree
		sub_tree = TreeNode(is_leaf = False,leaf_score = None,
				split_feature = best_feature,
				split_threshold = best_threshold,
				left_child = left_tree,\
				right_child = right_tree,
				nan_direction = nan_direction)

		return sub_tree

	def fit(self,X,Y,max_depth = 5,min_child_weight = 1,
			colsample_bylevel = 1.0,min_sample_split = 10,\
			reg_lambda = 1.0,gamma = 0.0,num_thread = -1):
		'''
		X:pd.DataFram [n_sampels,n_features]
		Y:pd.DataFram [n_samples,5],column is [label,y_pred,grad,hess,sample_weight]
		'''
		self.min_child_weight = min_child_weight
		self.colsample_bylevel = colsample_bylevel
		self.min_sample_split = min_sample_split
		self.reg_lambda = reg_lambda
		self.gamma = gamma
		self.num_thread = num_thread
		
		self.root = self.buildTree(X,Y,max_depth)

	def _predict(self,tree_node,X):
		'''
		predict a single sample
		note that X is a tupe(index,pandas.core.series.Series) from df.iterrows()
		'''
		if tree_node.is_leaf:
			return tree_node.leaf_score

		elif pd.isnull(X[1][tree_node.split_feature]):
			if tree_node.nan_direction == 0:
				return self._predict(tree_node.left_child,X)
			else:
				return self._predict(tree_node.right_child,X)
		elif X[1][tree_node.split_feature] < tree_node.split_threshold:
			return self._predict(tree_node.left_child,X)
		else:
			return self._predict(tree_node.right_child,X)
	
	def predict(self,X):
		'''
		predict multi samples
		X is DataFrame
		'''
		preds = None
		samples = X.iterrows()

		func = partial(self._predict,self.root)
		
		if self.num_thread == -1:
			pool = Pool()
			preds = pool.map(func,samples)
			pool.close()
			pool.join()
		else:
			pool = Pool(self.num_thread)
			preds = pool.map(func,samples)
			pool.close()
			pool.join()
		
		return np.array(preds)






















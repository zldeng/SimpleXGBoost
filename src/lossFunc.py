#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2017-11-24 10:01:37
'''

import abc
import sys
import numpy as np

from autograd import elementwise_grad

elementwise_hess = lambda func: elementwise_grad(elementwise_grad(func))

class BaseLoss(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self,reg_lambda):
		self.reg_lambda = reg_lambda
	
	@abc.abstractmethod
	def grad(self,preds,labels):
		pass

	@abc.abstractmethod
	def hess(self,preds,labels):
		pass
	
	@abc.abstractmethod
	def transform(self,preds):
		pass


class LogisticLoss(BaseLoss):
	'''
	label is {0,1}
	'''
	def transform(self,preds):
		return 1.0/(1.0 + np.exp(-preds))

	def grad(self,preds,labels):
		preds = self.transform(preds)

		return (1-labels)/(1-preds) - labels/preds

	def hess(self,preds,labels):
		preds = self.transform(preds)

		return labels / np.square(preds) + (1-labels) / np.square(1-preds)



class SquareLoss(BaseLoss):
	"""
	SquareLoss_l2regularization = SquareLoss(10)
	"""
	def transform(self, preds):
		return preds

	def grad(self, preds, labels):
		return preds-labels

	def hess(self, preds, labels):
		return np.ones_like(labels)

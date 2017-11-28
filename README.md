# SimpleXGBoost
XGBoost的简单实现。  
该代码只实现了XGBoost论文中使用的创建回归树的相关部分，支持自定义损失函数，但未进行性能优化。

# 文件说明
## src
### gboostTree.py
gboostTree.py实现单棵回归树的创建，在查找分裂点时使用多进行加快速度。
### gboost.py
gboost.py实现创建整个回归树序列。
### lossFunc.py
lossFunc.py实现了logisticLoss和SquareLoss，同时定义了损失函数的基类，任何自定义的继承自该基类的损失函数均可昨晚xgboost的损失函数。

## test
### binaryClassification.py
binaryClassification.py通过构造二元分类数据对代码进行测试

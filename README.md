# data_mining_G50
2024_spring_data_mining
##
Recursive feature elimination (RFECV) 

## 
在关联规则挖掘中，每个观测样本称为一个记录（transaction），每个记录是一组元素（item）的集合。
传统的关联规则挖掘方法不区分记录中元素的类别，传统方法可能挖掘出大量不具有较高应用价值的同类别关联规则
在产生初始词典的过程中限制主题的类别，对同类别的关联规则进行剪枝，能更准确、高效地挖掘跨类别的关联规则

##
MAE对异常点有更好的鲁棒性

MSE对误差取了平方，相对于使用MAE计算损失，使用MSE的模型会赋予异常点更大的权重

MAE存在一个严重的问题（特别是对于神经网络）：更新的梯度始终相同，也就是说，即使对于很小的损失值，梯度也很大。这样不利于模型的学习

## ranking 
RankNet、LambdaMART、RankBoost

### Stacking

#### LambdaMART



'''
Author: liubai
Date: 2021-03-08
LastEditTime: 2021-03-08
'''
# 使用pytorch实现softmax

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import d2l_pytorch as d2l


# 定义一个展平层
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

# 搭建模型
def linear_model(train_iter,test_iter,batch_size=256,num_epochs=5):
    # 模型输入和输出
    num_inputs=784
    num_outputs=10
    # 次数
    # batch_size=256
    # 小批量样本数
    # num_epochs=5
    
    # 定义模型，写法二
    # 使用nn.Sequential有序容器，俺早添加到计算图中顺序，搭建网络
    print('construct model net...')
    net = nn.Sequential()
    net.add_module('flatten',FlattenLayer())
    net.add_module('linear', nn.Linear(num_inputs, num_outputs))
    # 打印模型
    print('model net:');print(net)

    # 初始化模型参数
    print('init net.xx.weight...')
    init.normal_(net.linear.weight,mean=0,std=0.01)
    init.constant_(net.linear.bias,val=0)

    # 定义损失函数
    print('loss...')
    loss=nn.CrossEntropyLoss()

    # 定义优化算法
    print('optimizer...')
    optimizer=torch.optim.SGD(net.parameters(),lr=0.1)
    
    # 训练模型
    print('training ...')
    d2l.train_ch03(net, train_iter, test_iter, loss, batch_size,num_epochs, None, None, optimizer)

    # 返回net,此时的模型参数已经欸更新
    return net 

# 使用模型和学习到的参数，进行预测
def predict(net,X):
    # 返回向量，即批量预测
    # 每个样本（即每张图片），只有一个类别标签
    return net(X).argmax(dim=1).numpy()

def show_predict(net,x,y):
    
    labels=d2l.get_fashion_mnist_labels(y.numpy())
    pred=d2l.get_fashion_mnist_labels(predict(net,x))

    # titles = [true + '\n' + pred for true, pred in zip(labels, pred)]
    # d2l.show_fashion_mnist(x[0:9],titles)
    for true, pred in zip(labels, pred):
        print(' label  :%s\n predict:%s'%(true, pred))


if __name__=='__main__':

    # 读取训练数据和测试数据
    batch_size=256
    print('load data ...')
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

    # 训练模型
    net=linear_model(train_iter,test_iter)

    # 预测
    x,y=iter(test_iter).next()
    show_predict(net,x,y)





    
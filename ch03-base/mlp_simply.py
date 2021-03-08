'''
Author: liubai
Date: 2021-03-08
LastEditTime: 2021-03-08
'''
'''
Author: liubai
Date: 2021-03-08
LastEditTime: 2021-03-08
'''
# 利用torch包，简洁实现MLP

import torch
import torch.nn as nn

import numpy as np
import d2l_pytorch as d2l
import sys

def MLP():
    # 输入层，隐藏层，输出层
    num_inputs,num_outputs,num_hiddens=784,10,256
    
    # 定义模型
    net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )
    # 模型参数，初始化
    for param in net.parameters():
        nn.init.normal_(param,mean=0,std=0.01)
    
    # load data
    batch_size=256
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

    # loss
    loss=torch.nn.CrossEntropyLoss()

    # 更新参数
    optimizer=torch.optim.SGD(net.parameters(),lr=0.5)

    # training model
    num_epochs=5
    print('traning ...')
    d2l.train_ch03(net,train_iter,test_iter,loss,batch_size,num_epochs,None,None,optimizer)

    # 返回模型，此时参数已经被更新
    return net

if __name__=='__main__':

    # load data
    # batch_size=256
    # train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

    net=MLP()

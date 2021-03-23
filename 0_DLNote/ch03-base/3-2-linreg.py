'''
Author: liubai
Date: 2021-03-05
LastEditTime: 2021-03-05
'''

import torch
import matplotlib.pyplot as plt 
import numpy as np
import random

num_feature=2
num_example=1000
true_w=[2,-3.4]
true_b=4.2

def loadData():
    dataset=torch.randn(num_example,num_feature,
        dtype=torch.float32)
    labels=true_w[0]*dataset[:,0]+true_w[1]*dataset[:,1]+true_b

    # 给label，添加一些随机噪声：服从均值为0，标准差为0.01的正态分布
    labels+=torch.tensor(np.random.normal(0,0.01),
        dtype=torch.float32)
    return dataset,labels


def show(dataset,labels):
    plt.figure()
    # dataset为torch.tensor类型，
    # 转为numpy类型
    # 画出第2个feature和labels之间的关系图
    plt.scatter(dataset[:,1].numpy(),labels.numpy(),1)
    plt.show()


def data_iter(batch_size,dataset,labels):
    # 返回batch_size个样本
    num_example=len(dataset)
    idxs=list(range(num_example))
    # 随机选择
    random.shuffle(idxs)
    # 得到batch_size个样本
    # 下次从yield的位置，继续循环，取batch_size个
    for i in range(0,num_example,batch_size):
        end=min(i+batch_size,num_example)
        j=torch.LongTensor(idxs[i:end])
        # print("j:",j)
        # dataset.index_select(0,j)中的0表示按行索引
        yield dataset.index_select(0,j),labels.index_select(0,j)

def linreg(X,w,b):
    # 计算预测值
    # 矩阵运算torch.mm()
    return torch.mm(X,w)+b

def squared_loss(y_hat,y):
    # 计算损失:平方差损失
    return (y_hat-y.view(y_hat.size()))**2/2

# 定义优化函数
def sgd(params,lr,batch_size):
    # prams为[w,b],
    # lr为超参数学习率
    for param in params:
        # 使用梯度,直接更改[w,b]参数
        # 这里要使用.data
        param.data-=lr*param.grad/batch_size

def model():
    # hyperparments
    # learning rate
    lr=0.03
    # iter period
    num_epochs=3
    # batch
    batch_size=10

    # net
    net=linreg
    # loss function
    loss=squared_loss
    
    # init params
    w=torch.tensor(np.random.normal(0,0.01,size=(num_feature,1)),
        dtype=torch.float32,requires_grad=True)
    b=torch.zeros(1,dtype=torch.float32,requires_grad=True)
    

    # 一共迭代iterm次数
    for epoch in range(num_epochs):
        # 每一个迭代周期内，都会使用训练集中所有的样本依次
        for X,y in data_iter(batch_size,dataset,labels):
            # 损失l是小批量X，y的损失
            l=loss(net(X,w,b),y).sum()
            # 小批量损失，对模型参数求梯度
            l.backward()
            # 使用小批量随机梯度下降，迭代模型参数
            sgd([w,b],lr,batch_size)

            # 梯度清零，防止累加
            w.grad.data.zero_()
            b.grad.data.zero_()
        # 计算一次迭代周期完毕之后的损失值
        train_loss=loss(net(dataset,w,b),labels)
        # tensormean().item()是将一个tensor标量，转为python数值
        print('epoch %d, loss %f'%(epoch+1,train_loss.mean().item()))
    
    # 迭代完成，return w,b
    return w,b

if __name__=='__main__':
    
    # load data
    dataset,labels=loadData()
    # show(dataset,labels)
    
    # 模型
    w,b=model()  # 带有梯度信息
    print("w.T:",w.T)
    print("b:",b)
    
    # print squared err
    err=torch.sqrt((torch.tensor(true_w)-w.T)**2).sum()
    print("error:%.5f"%err.item())
    
   

'''
Author: liubai
Date: 2021-03-05
LastEditTime: 2021-03-05
'''

import torch
import numpy as np
import torch.utils.data as Data  #加载数据
import torch.nn as nn   # 定义网络，nn.init初始化网络
import torch.optim as optim  # 优化器


num_feature=2
num_example=1000
true_w=[2,-3.4]
true_b=4.2

def loadData():
    dataArr=torch.randn(num_example,num_feature,
        dtype=torch.float)
    labels=true_w[0]*dataArr[:,0]+true_w[1]*dataArr[:,1]+true_b

    # 给label，添加一些随机噪声：服从均值为0，标准差为0.01的正态分布
    labels+=torch.tensor(np.random.normal(0,0.01),
        dtype=torch.float)
    return dataArr,labels


def model(dataArr,labels):
    # 一次读取batch_size个数据
    batch_size=10

    # 组合数据特征和标签
    dataset=Data.TensorDataset(dataArr,labels)

    # 随机读取批量数据
    data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)

    # net
    net=nn.Sequential( nn.Linear(num_feature,1)) 
    
    # 初始化权重为：均值为0，方差为0.01的正态分布
    nn.init.normal_(net[0].weight,mean=0,std=0.01)
    nn.init.constant(net[0].bias,val=0)
    
    # 定义损失函数
    loss=nn.MSELoss()

    # 定义优化算法
    # ls为学习率
    optimizer=optim.SGD(net.parameters(),lr=0.03)

    # 训练模型
    num_epoch=3
    for epoch in range(1,num_epoch+1):
        for X,y in data_iter:
            out=net(X)
            l=loss(out,y.view(-1,1))
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            # optim.step函数，用来迭代模型参数，对批量中样本梯度求平均
            optimizer.step()
        print('epoch %d, loss:%f' % (epoch,l.item()))

    # return weights
    dense=net[0]
    return dense

if __name__=='__main__':

    # 加载数据
    dataArr,labels=loadData()

    # 训练得到的权重w和b
    dense=model(dataArr,labels)
    print(dense.weight)
    print(dense.bias)
    
    # 计算w的均方差
    w=dense.weight
    err=torch.sqrt((torch.tensor(true_w)-w)**2).sum()
    print("error:%.5f"%err.item())





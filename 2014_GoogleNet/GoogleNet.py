'''
Author: liubai
Date: 2021-03-26
LastEditTime: 2021-03-26
'''

import torch
from torch import nn,optim
import torch.nn.functional as F 

import numpy as np 
import time 
# import sys
# sys.path.append('d:/Github/DIDL/')
import os

from script import d2l_pytorch as d2l 

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Inception(nn.Module):
    # 4条线路
    def __init__(self,in_c,c1,c2,c3,c4):
        super(Inception,self).__init__()
        # 线路1，共1层，就是最左侧的那条
        ## 1*1的卷积层，用来减少通道数
        self.p1_1=nn.Conv2d(in_channels=in_c,out_channels=c1,kernel_size=1)
        # 线路2，共2层
        ## 1*1的卷积层
        self.p2_1=nn.Conv2d(in_channels=in_c,out_channels=c2[0],kernel_size=1)
        ## 3*3的卷积层
        self.p2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        # 线路3，共2层
        ## 1*1的卷积层
        self.p3_1=nn.Conv2d(in_c,c3[0],kernel_size=1)
        ## 5*5的卷积层
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        # 线路4，共2层
        ## 3*3的最大池化层
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_c,c4,kernel_size=1)

    def forward(self,x):
        # 线路1
        p1=F.relu(self.p1_1(x))
        # 线路2
        p2=F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # 线路3
        p3=F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # 线路4
        p4=F.relu(self.p4_2(self.p4_1(x)))
        # 将四条线路的输出，在通道维上连结
        return torch.cat((p1,p2,p3,p4),dim=1) 

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()

        self.b1=nn.Sequential(
        nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b2=nn.Sequential(
        # 1*1的卷积层
        nn.Conv2d(64,64,kernel_size=1),
        # 3*3的卷积层，将通道数增加3倍
        nn.Conv2d(64,192,kernel_size=3,padding=1),
        # 池化层
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b3=nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            d2l.GlobalAvgPool2d()
        )
        self.final=nn.Sequential(
            d2l.FlattenLayer(), 
            nn.Linear(1024, 10)
        )
    def forward(self,x):
        x=self.b1(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b4(x)
        x=self.b5(x)
        x=self.final(x)
        return x


net=GoogleNet()
print(net)

# test net
X=torch.rand(1,1,96,96)
for blk in net.children():
    X=blk(X)
    print('out shape:',X.shape)



# 设置超参数

batch_size=64
lr,num_epochs=0.001,5
optimizer=torch.optim.Adam(net.parameters(),lr=lr)
# 交叉熵损失函数
loss=torch.nn.CrossEntropyLoss()


# 是否加载预训练权重文件

pretrained_weights = None

if pretrained_weights != None:
    net.load_state_dict(torch.load(pretrained_weights))
else:
    net.init_weights()



# 获取训练集，测试集
train_iter,test_iter=d2l.load_data_fashion_mnist_resize(batch_size,resize=96)

num_classes=10

# 记录 loss, acc 变化
# 
train_loss_list = list()
train_accuracy_list = list()
test_loss_list = list()
test_accuracy_list = list()



net=net.to(device)
print('training on... ',device)
 
for epoch in range(num_epochs):
    start=time.time()
    
    net.train()
    
    train_l_sum,train_acc_sum,n,batch_count=0.0,0.0,0,0
    
    for i,(X,y) in enumerate(train_iter):
        X=X.to(device)
        y=y.to(device)
        y_hat=net(X)
        # 计算损失
        l=loss(y_hat,y)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        l.backward()
        # 更新参数
        optimizer.step()
        print('epoch %d/%d, iter %d/391, loss %.3f' % (epoch,num_epochs,i,l.cpu().item()))
        if (i+1) % 10 == 0:
            # print('epoch %d/%d, iter %d/391, loss %.3f' % (epoch,num_epochs,i,l.cpu().item()))
            print('label(GT):',y[:10].numpy())
            print('label(PD):',np.argmax(y_hat.cpu().data.numpy()[:10], axis=1))
        
        # 更新损失和正确率
        train_l_sum+=l.cpu().item()
        train_acc_sum+=(y_hat.argmax(dim=1) == y ).sum().cpu().item()
        n+=y.shape[0]
        batch_count+=1

        # test net
        if (i+1) % 5==0:
            break

    # 计算train_loss, train_acc,test_acc
    train_loss=train_l_sum/batch_count
    train_acc=train_acc_sum/n
    # 测试集上的正确率
    test_acc,test_loss=d2l.eval_acc_loss(test_iter,net,loss)
    
    # 保持 acc,loss
    print('epoch:',epoch+1, 'train loss %.2f' % (train_loss), 'train acc %.3f%%' % (train_acc*100), 'test loss %.2f' % (test_loss), 'test acc %.3f%%' % (test_acc*100))
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_acc)
    test_accuracy_list.append(test_acc)
    test_loss_list.append(test_loss)
    print('epoch',epoch, 'time elapsed %.1f'%(time.time()-start) )
    
    # save weights
    if not os.path.isdir('./weights'):
        os.mkdir('weights')
    torch.save(net.state_dict(), './weights/pretrained_weights_{}.pth'.format(epoch+1))

print('training finish...')


import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')  
# 以矢量图方式，保存图片

# 画出 loss, acc 变化图

x = np.arange(len(train_accuracy_list) + 1)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1)
plt.plot(x, [0] + train_accuracy_list)
plt.plot(x, [0] + test_accuracy_list)
plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
plt.grid(True)


plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(x, train_loss_list[0:1] + train_loss_list)
plt.plot(x, train_loss_list[0:1] + test_loss_list)
plt.legend(['training loss', 'testing loss'], loc='upper right')
plt.savefig('loss_numCls{}_epoch{}.png'.format(num_classes, epoch+1))
plt.show() 


'''
Author: liubai
Date: 2021-03-08
LastEditTime: 2021-03-08
'''


import torch
import torchvision

# load_data_fashion_mnist()
import torchvision.transforms as transforms  

import matplotlib.pyplot as plt
from IPython import display
import time
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch

# 加载fashionMNIST数据
def load_data_fashion_mnist(batch_size):    
    # fashionMNIST
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../', 
        train=True, download=True,                   
        transform=transforms.ToTensor())

    mnist_test = torchvision.datasets.FashionMNIST(
        root='../', 
        train=False, download=True,         
        transform=transforms.ToTensor())
        
    # 取batch_size个样本数据
    # 多进程取样本数据 
    if sys.platform.startswith('win'):
        # 进程数
        num_workers=0
    else:
        num_workers=4
    
    # 随机选取batch_size个数据样本
    train_iter=torch.utils.data.DataLoader(mnist_train,
        batch_size=batch_size,
        shuffle=True,num_workers=num_workers)
    test_iter=torch.utils.data.DataLoader(mnist_test,
        batch_size=batch_size,shuffle=False,num_workers=num_workers)
    
    return train_iter,test_iter


# 定义函数，将数值类别转为文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover',               'dress', 'coat','sandal', 'shirt',                      'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 定义一个函数，在一行中，画出多个图像及其标签
def show_fashion_mnist(images, labels):
    # d2l.use_svg_display()替换为display.set_matplotlib_formats('svg')
    # 载入一批图片，以SVG格式显示图片
    display.set_matplotlib_formats('svg')
    
    # plt.subplots()返回(figure,axes)
    # 等价于fig = plt.figure();fig.add_subplt(111)
    # 这里的_表示我们忽略（不使用）的变量
    # 将figure分为 1*len(images)个子图
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    # for xx in zip(yy)表示并行遍历
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# 评估net模型在data_iter小批量数据上的准确率
def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    for X,y in data_iter:
        acc_sum+=(net(X).argmax(dim=1) == y).float().sum().item()
        n+=y.shape[0]
    return acc_sum/n

# 反向传播，更新参数params
def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，
    # 但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size 
        # 注意这里更改param时用的param.data

# 定义函数，训练模型
def train_ch03(net,train_iter,test_iter,loss,batch_size,num_epochs,params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_loss_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            # 预测值
            y_hat=net(X)
            # 计算损失
            l=loss(y_hat,y).sum()

            # 梯度反传之前，梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            # 梯度反传
            l.backward()

            if optimizer is None:
                # 更新参数
                sgd(params,lr,batch_size)
            else:
                optimizer.step()
            
            # 累加损失
            # 累加错误个数
            # 累加样本个数
            train_loss_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1) == y ).sum().item()
            n+=y.shape[0]
        # 一遍epoch结束，计算在测试集上的准确率
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f,test acc %.3f'
                %(epoch+1,train_loss_sum/n,train_acc_sum/n,test_acc))

# 定义一个展平层
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

    

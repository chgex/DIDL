'''
Author: liubai
Date: 2021-03-08
LastEditTime: 2021-03-25
'''


import torch
import torchvision
from torch import nn,optim

import torchvision.transforms as transforms  

import matplotlib.pyplot as plt
from IPython import display
import time
import sys
sys.path.append("..") # 为了导入上层目录script中的d2lzh_pytorch





# 加载fashionMNIST数据
def load_data_fashion_mnist(batch_size):    
    # fashionMNIST
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../datasets/', 
        train=True, download=False,                   
        transform=transforms.ToTensor())

    mnist_test = torchvision.datasets.FashionMNIST(
        root='../datasets/', 
        train=False, download=False,         
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


# 反向传播，更新参数params
def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，
    # 但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size 
        # 注意这里更改param时用的param.data


# 定义一个展平层
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)



def evaluate_accuracy(data_iter,net,device=None):
    # gpu
    if device is None and isinstance(net,torch.nn.Module):
        # 如果没有指定device，则使用net的device
        device=list(net.parameters())[0].device
    # 准确率，总数
    acc_sum,n=0.0,0
    # with torch.no_grad： disables tracking of gradients in autograd. 
    # model.eval()： changes the forward() behaviour of the module it is called upon.
    with torch.no_grad():
        # for X,y in data_iter:
        for i,(X,y) in enumerate(data_iter):
            if isinstance(net,torch.nn.Module):
                # 评估模式，该模式会关闭dropout
                net.eval()
                # torch.argmax(input, dim, keepdim=False) → LongTensor返回指定维度的最大值的索引。
                acc_sum+=( net(X.to(device)).argmax(dim=1) == y.to(device) ).float().sum().cpu().item()
            else: # 无GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n+=y.shape[0]
            if (i+1)%5==0:
                break
    return acc_sum/n



def load_data_fashion_mnist_resize(batch_size, resize=None, root='../datasets'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def train(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    net=net.to(device)
    print('training on ',device)
    # 损失函数，使用交叉熵损失函数
    loss=torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,batch_count,start=0.0,0.0,0,0,time.time()
        for i,(X,y) in enumerate(train_iter):
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            # 更新参数
            optimizer.step()
            
            print('epoch %d/%d, iter %d/%d, loss %.3f' 
                    % (epoch,num_epochs,i,60000//batch_size,l.cpu().item()))

            # 更新损失和正确率
            train_l_sum+=l.cpu().item()
            train_acc_sum+=(y_hat.argmax(dim=1) == y ).sum().cpu().item()
            n+=y.shape[0]
            batch_count+=1
        # 测试集上的正确率
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
        %(epoch+1,train_l_sum/batch_count,train_acc_sum/n,test_acc,time.time()-start))    


# 定义全局池化层
import torch.nn.functional as F 

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()
    def forward(self,x):
        # 全局池化层的大小，等于输入的高和宽
        return F.avg_pool2d(x,kernel_size=x.size()[2:])




def load_data_CIFAR10(batch_size):
    # 加载数据集

    # 图像预处理
    # 构建一个列表：将多个图像变换实例集合到一起
    resize=227
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(size=resize),
        transforms.ToTensor(),
        # Normalize 这8个值是针对 CIFAR-10 这个数据集算出来的，对于其他数据集不适用
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # CIFAR10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../datasets/', train=True, 
                    download=False, transform=transform)

    test_dataset= torchvision.datasets.CIFAR10(root='../datasets/', train=False,
        download=False, transform=transform)

    # data loader
    train_iter=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4)
    test_iter=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4)
    
    return train_iter,test_iter



def eval_acc_loss(data_iter,net,loss,device=None):
    # gpu
    if device is None and isinstance(net,torch.nn.Module):
        # 如果没有指定device，则使用net的device
        device=list(net.parameters())[0].device
    
    loss_sum,acc_sum,n,batch_count=0.0,0.0,0,0

    # with torch.no_grad： disables tracking of gradients in autograd. 
    # model.eval()： changes the forward() behaviour of the module it is called upon.
    with torch.no_grad():
        # for X,y in data_iter:
        for i,(X,y) in enumerate(data_iter):
            # 评估模式，该模式会关闭dropout
            net.eval()
            # torch.argmax(input, dim, keepdim=False) → LongTensor返回指定维度的最大值的索引。
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            # 计算损失
            l=loss(y_hat,y)
            
            loss_sum+=l.cpu().item()
            acc_sum+=(y_hat.argmax(dim=1) == y ).sum().cpu().item()
            n+=y.shape[0]
            batch_count+=1

            if (i+1)%5==0:  break
        
        test_loss=loss_sum/batch_count
        test_acc=acc_sum/n       
            
    return test_acc,test_loss

'''
Author: liubai
Date: 2021-04-01
LastEditTime: 2021-04-01
'''
import torch
import cv2
import numpy as np 

import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models import alexnet 

import time

# 自定义脚本
import utils


# 转换推荐区域的尺寸
# 将proposal region的大小，转为227*227
def get_transform():
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        # transforms.Pad(padding=True,padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


# 选择性搜索算法
def selectSearch(img_path): 
    image=cv2.imread(img_path,cv2.IMREAD_COLOR)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    # ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    
    return rects



def get_model(device=None):
    # 加载CNN模型
    model = alexnet(pretrained=True)
    
    num_classes = 1000
    model.classifier[6].out_features=num_classes
    
    # 直接使用alexnet在ImageNet上预训练权重
    # model.load_state_dict(torch.load('./backup/weights.pth'))
    
    model.eval()
    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model

# 计算交并比：IOU
def iou(rect_a, rect_b):
    # rect_a: [x,y,w,h]
    a_w,a_h=rect_a[2],rect_a[3]
    b_w,b_h=rect_b[2],rect_b[3]
    
    # x,y,w,h to xmin,ymin,xmax,ymax  
    a_xmin,a_ymin,a_xmax,a_ymax=utils.wh2box(rect_a) #[0]
    b_xmin,b_ymin,b_xmax,b_ymax=utils.wh2box(rect_b)
    
    # iou_w,iou_h
    iou_w=min(a_xmin,a_xmax,b_xmin,b_xmax)+a_w+b_w-max(a_xmin,a_xmax,b_xmin,b_xmax)
    iou_h=min(a_ymin,a_ymax,b_ymin,b_ymax)+a_h+b_h-max(a_ymin,a_ymax,b_ymin,b_ymax)

    # 计算交并比
    iou_ratio=0.0
    if iou_w <=0 or iou_w<=0:
        # 无交集
        iou_ratio=0.0
    else:
        # 计算交集面积
        intersection = iou_w * iou_h
        # 计算两个边界框的面积
        a_area=a_w * a_h
        b_area=b_w * b_h
        # 交并比
        iou_ratio = intersection / (a_area + b_area - intersection)
    return iou_ratio

# 非极大值抑制
def nms(rect_list, score_list):

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    # 按照p[idx]的概率得分score, 进行排序，记住对应的idx
    # 使用[::-1], 达到倒序效果
    idxs = np.argsort(score_array)[::-1]
    print('len(idxs)',len(idxs))
    
    nms_rects = list()
    nms_scores = list()

    thresh = 0.2
    while len(idxs)>0:
        
        idx=idxs[0]
        
        # 添加scores最大的边界框
        nms_rects.append(rect_array[idx])
        nms_scores.append(score_array[idx])
        
        # 如果这是最后一个边界框
        num=len(idxs)
        if num - 1 == 0 : 
            break
        print('num: ',num)

        tmp_idxs=[]

        # 计算IoU，只保留交并比小于阈值的边界框
        tmp_idxs=[]
        for i in idxs[1:]:
            iou_ratio = iou(rect_array[idx],rect_array[i])
            if iou_ratio < thresh:
                tmp_idxs.append(i)
        idxs=np.array(tmp_idxs)
    return nms_rects, nms_scores




if __name__=='__main__':

    img_1='./000012.jpg'
    xml_1='./000012.xml'

    # get coordinate of groung truth bounding box
    gtbox=utils.parse_xml(xml_1)
    # show ground truth
    utils.drow_gtbox(img_1,gtbox)

    # 加载模型
    model=get_model()
    print('model construct:',model)

    # 选择性搜索，生成推荐区域
    rects=selectSearch(img_1)
    print('selectsearch has find %d regions' %(len(rects)))


    start = time.time()
    # 保存正样本box及其得分
    positive_list =[]
    score_list = []

    # 1000个类别中，小汽车对应的idx
    idx=713

    svm_thresh=0.4   #0.4  没有经过训练的softmax线性层，输出的ground_truth对应的prob[1]=7.73,softmax之后，没有进行归一化处理

    trans=get_transform()

    image=cv2.imread(img_1,cv2.IMREAD_COLOR)

    for i,rect in enumerate(rects):
        
        bbox=utils.wh2box(rect)   # rect: x,y,w,h
        xmin,ymin,xmax,ymax=bbox
        
        # 根据坐标，获得region
        region = image[ymin:ymax,xmin:xmax]
        
        # 将region大小转为227*227
        img = trans(region)
        
        # 使用CNNs，提取出特征向量
        output = model(img.unsqueeze(0))[0]
        
        if i % 150 ==0:    print('%d / 1532' % (i) )
        # output.shape为(1000,)
        if torch.argmax(output).item() == idx:  
            # 预测为idx
            probs = torch.softmax(output, dim=0).cpu().numpy()
            if probs[idx] > svm_thresh:
                score_list.append(probs[idx])
                positive_list.append(rect)
    end=time.time()
    print('CNNs time span ',end-start)
    print( '%d region with possible class idx '  % (len(score_list)))


    bbox_list,prob_list= nms(positive_list, score_list)
    print('after nms, remain %d regions' % (len(bbox_list)))

    utils.draw_bbox(img_1,bbox_list,prob_list)

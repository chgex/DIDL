'''
Author: liubai
Date: 2021-04-01
LastEditTime: 2021-04-01
'''
import xmltodict
import numpy as np  
import matplotlib.pyplot as plt
import random

import cv2

# xml --> xmin,ymin,xmax,ymax
def parse_xml(xml_path):
    """
    解析xml文件，返回标注边界框坐标
    """
    # print(xml_path)
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        # print(xml_dict)

        bndboxs = list()
        objects = xml_dict['annotation']['object']
        if isinstance(objects, list):
            for obj in objects:
                obj_name = obj['name']
                difficult = int(obj['difficult'])
                if 'car'.__eq__(obj_name) and difficult != 1:
                    bndbox = obj['bndbox']
                    bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        elif isinstance(objects, dict):
            obj_name = objects['name']
            difficult = int(objects['difficult'])
            if 'car'.__eq__(obj_name) and difficult != 1:
                bndbox = objects['bndbox']
                bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        else:
            pass

        return np.array(bndboxs)[0]


# 图像上显示ground truth bounding box
def drow_gtbox(img_path,gtbox,text='ground_truth'):
    # cv2 读取图像
    image=cv2.imread(img_path,cv2.IMREAD_COLOR)
    # bounding box
    xmin,ymin,xmax,ymax=gtbox
    
    # 画边界框
    cv2.rectangle(image, (xmin,ymin), (xmax, ymax), (0,0,255), thickness=0)
    
    # 标注文本‘groung_truth’
    cv2.putText(image,text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    # 显示图像
    # tmp=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(tmp)
    cv2.imshow('gtbox',image)
    cv2.waitKey(0)


# selectsearch算法返回的坐标为x,y,w,h
# cv2.rectangle使用xmin,ymin,xmax,ymax坐标
# 所以定义坐标转换函数
 
def get_wh(gtbox):
    # xmin,ymin,xmax,ymax --->  x,y,w,h
    xmin,ymin,xmax,ymax=gtbox[0]
    x=min(xmin,xmax);y=min(ymin,ymax)
    w=abs(xmax-xmin);h=abs(ymax-ymin)
    return x,y,w,h

def wh2box(box):
    # x,y,w,h ---> xmin,ymin,xmax,ymax
    x,y,w,h=box
    xmin=x;ymin=y
    xmax=x+w;ymax=y+h
    return np.array([xmin,ymin, xmax, ymax])


def draw_bbox(image_path, bbox_list, prob_list):
    # 读取 图片
    img=cv2.imread(image_path,cv2.IMREAD_COLOR)
    for i,rect in enumerate(bbox_list):
        x,y,w,h=rect
        # bbox=wh2box(rect)   # rect: x,y,w,h
        # xmin,ymin,xmax,ymax=bbox[0]
        prob = prob_list[i]

        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=0)
        cv2.putText(img, '%.2f' % (prob), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # plt.title('bounding boxs')
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
    cv2.imshow('bounding boxs',img)
    cv2.waitKey(0)
    
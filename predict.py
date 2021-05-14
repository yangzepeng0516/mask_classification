import os
import zipfile
import random
import json
import paddle
import sys
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import paddle.fluid as fluid
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from model import VGGNet
from data_processor import *



def load_image(img_path):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = img / 255  # 像素值归一化
    return img


label_dic = train_parameters['label_dict']

'''
模型预测
'''
with fluid.dygraph.guard():
    model, _ = fluid.dygraph.load_dygraph("vgg")
    vgg = VGGNet()
    vgg.load_dict(model)
    vgg.eval()

    # 展示预测图片
    infer_path = '/home/aistudio/data/data23615/infer_mask01.jpg'
    img = Image.open(infer_path)
    plt.imshow(img)  # 根据数组绘制图像
    plt.show()  # 显示图像

    # 对预测图片进行预处理
    infer_imgs = []
    infer_imgs.append(load_image(infer_path))
    infer_imgs = np.array(infer_imgs)

    for i in range(len(infer_imgs)):
        data = infer_imgs[i]
        dy_x_data = np.array(data).astype('float32')
        dy_x_data = dy_x_data[np.newaxis, :, :, :]
        img = fluid.dygraph.to_variable(dy_x_data)
        out = vgg(img)
        lab = np.argmax(out.numpy())  # argmax():返回最大数的索引
        print("第{}个样本,被预测为：{}".format(i + 1, label_dic[str(lab)]))

print("结束")
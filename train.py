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
from data_processor import train_parameters
from data_processor import *
from model import VGGNet

'''
模型训练
'''
# with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
with fluid.dygraph.guard():
    print(train_parameters['class_dim'])
    print(train_parameters['label_dict'])
    vgg = VGGNet()
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=train_parameters['learning_strategy']['lr'],
                                              parameter_list=vgg.parameters())
    for epoch_num in range(train_parameters['num_epochs']):
        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([x[1] for x in data]).astype('int64')
            y_data = y_data[:, np.newaxis]

            # 将Numpy转换为DyGraph接收的输入
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)

            out, acc = vgg(img, label)
            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)

            # 使用backward()方法可以执行反向网络
            avg_loss.backward()
            optimizer.minimize(avg_loss)

            # 将参数梯度清零以保证下一轮训练的正确性
            vgg.clear_gradients()

            all_train_iter = all_train_iter + train_parameters['train_batch_size']
            all_train_iters.append(all_train_iter)
            all_train_costs.append(loss.numpy()[0])
            all_train_accs.append(acc.numpy()[0])

            if batch_id % 1 == 0:
                print(
                    "Loss at epoch {} step {}: {}, acc: {}".format(epoch_num, batch_id, avg_loss.numpy(), acc.numpy()))

    draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")
    draw_process("trainning loss", "red", all_train_iters, all_train_costs, "trainning loss")
    draw_process("trainning acc", "green", all_train_iters, all_train_accs, "trainning acc")

    # 保存模型参数
    fluid.save_dygraph(vgg.state_dict(), "vgg")
    print("Final loss: {}".format(avg_loss.numpy()))

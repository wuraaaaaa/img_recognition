#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os,cv2,glob
import sys,argparse
import matplotlib.pyplot as plt
import time


# In[4]:


image_size=128
num_channels=3
images=[]
result_label=[]
dir_label=[]
dir_name=[]
fault_file_name=[]
path="./data/12/"
train_path = './data/dc_data/'
def get_cls(train_path):
    classes=[]
    glob_path = train_path + '*'
    full_files = sorted(glob.glob(glob_path))
    for i in range(len(full_files)):
        s_f = full_files[i].split(sep='\\')
        classes.append(s_f[-1])
    print('类别： ' + str(classes))
    return classes
classes=get_cls(train_path)
#print('类别： ' + str(classes))

def show_pic(img,name,top):
    plt.figure('show_img')
    #img=np.multiply(img,255.0)
   # print(img)
    plt.imshow(img)
    plt.axis('off')
    plt.title('top1 cls: '+name + ' PR: ' + str(round(top,4))) # 图像题目
    plt.text(30, 70, name,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})  # 显示在图片上
    plt.show()

    

def get_files(file_name):
    dir_name.append(file_name)
    
    name = file_name.split(sep='.') 
    #print(name[0])# 因为照片的格式是cat.1.jpg/cat.2.jpg
    if name[0] == 'dog':            # 所以只用读取 . 前面这个字符串
        dir_label.append(1)
    else:
        dir_label.append(0)
def totla_file():
    cat_num=0
    dog_num=0
    for i in range(len(dir_label)):
        if dir_label[i] == 1:
            cat_num+=1
        else:
            dog_num+=1
    #print(dir_label)
    print("狗有 %d 个，猫有 %d 个"%(dog_num,cat_num))

def find_fault(result,original):
    fault_num=0
    
    fault_label=[]
    for i in range(len(original)):
        if result[i] != original[i]:
            fault_num+=1
            fault_label.append(1)
        else:
            fault_label.append(0)
    correct_num = len(original) - fault_num
    pricent = correct_num / len(original)
    return pricent , fault_label



direct=os.listdir(path)
for file in direct:
    image=cv2.imread(path+file)
    #print("adress:",path+file)
    get_files(file)
    image=cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
    images.append(image) 
totla_file()

images=np.array(images,dtype=np.float32)
check_img = images
images=images.astype('float32')

images=np.multiply(images,1.0/255.0)
sess=tf.Session()
saver=tf.train.import_meta_graph('./model_25k/dog-cat.ckpt-118849.meta')
saver.restore(sess,'./model_25k/dog-cat.ckpt-118849')
check_num = 0
start_time = time.time()
for img in images:
    
    
    x_batch=img.reshape(1,image_size,image_size,num_channels)
   
    #sess=tf.Session()

#step1网络结构图
    #saver=tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-7496.meta')

#step2加载权重参数
    #saver.restore(sess,'./dogs-cats-model/dog-cat.ckpt-7496')

#获取默认的图
    graph=tf.get_default_graph()

    y_pred=graph.get_tensor_by_name("y_pred:0")

    x=graph.get_tensor_by_name("x:0")
    y_true=graph.get_tensor_by_name("y_true:0")
    y_test_images=np.zeros((1,2))

    feed_dict_testing={x:x_batch,y_true:y_test_images}
    result=sess.run(y_pred,feed_dict_testing)
   # res_label=['dog','cat']
    res_label=classes
    result_num=result.argmax()
######################################################
    show_pic(img,res_label[result_num],result[0][result_num])
    #print(res_label[result_num])
   # print(img)
#######################################################
    #check_num+=1
    #time.sleep(1)
    result_label.append (result_num )


end_time = time.time()
print('耗时：  ' + str(end_time - start_time) + 's')
pre_corr , fault = find_fault(result_label,dir_label)
#print(pre_corr,fault)
for i in range(len(result_label)):
    if fault[i]==1:
        fault_file_name.append(dir_name[i])
    
        
print("正确率为: " + str(round(pre_corr,5)*100) + '%')
print("错误的文件名为： ",fault_file_name)


# In[ ]:





# In[ ]:





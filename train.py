#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dataset
import glob
import tensorflow as tf
import numpy as np
from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(20)


def get_cls(train_path):
    classes=[]
    glob_path = train_path + '*'
    full_files = sorted(glob.glob(glob_path))
    for i in range(len(full_files)):
        s_f = full_files[i].split(sep='\\')
        classes.append(s_f[-1])
    return classes
train_path='./data/train_25k/'
learning_rate = 0.0001

batch_size=32
img_size=128
classes=get_cls(train_path)
print('准备训练的类别为'+ str(classes))
num_classes=len(classes)

validation_size=0.03

num_channels=3

data=dataset.read_train_sets(train_path,img_size,classes,validation_size)
print("读取数据完成")
#print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))

session=tf.Session()
x=tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name='x')
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)

filter_size_conv1=7
num_filters_conv1=64
conv_strides1 = 2
pool_strides1 = 2
pool_ksize1 = 2
use_pool1 = False  #(224-3)/1+1 = 222

filter_size_conv2=5
num_filters_conv2=128
conv_strides2 = 2
pool_strides2 = 2
pool_ksize2 = 2
use_pool2 = True #(222-3)/1+1=220  （220-2）/2 +1= 110

filter_size_conv3=5
num_filters_conv3=128
conv_strides3 = 1
pool_strides3 = 2
pool_ksize3 = 2
use_pool3 = False #(110-3)/1+1=108/2 =54

filter_size_conv4=3
num_filters_conv4=256
conv_strides4 = 1
pool_strides4 = 2
pool_ksize4 = 2
use_pool4 = True #(54-3)/1+1=52/2=26

filter_size_conv5=3
num_filters_conv5=512
conv_strides5 = 1
pool_strides5 = 2
pool_ksize5 = 2
use_pool5 = False #(26-3)/1+1=24

filter_size_conv6=3
num_filters_conv6=640
conv_strides6 = 1
pool_strides6 = 2
pool_ksize6 = 2
use_pool6 = False #22

filter_size_conv7=3
num_filters_conv7=512
conv_strides7 = 1
pool_strides7 = 2
pool_ksize7 = 2
use_pool7 = True #10

filter_size_conv8=3
num_filters_conv8=512
conv_strides8 = 1
pool_strides8 = 2
pool_ksize8 = 2
use_pool8 = False #8 

filter_size_conv9=3
num_filters_conv9=512
conv_strides9 = 1
pool_strides9 = 2
pool_ksize9 = 2
use_pool9 = False #3


#全连接层的输出
fc_layer_size=4096
fc_layer_size2 = 4096


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolution_layer(input,
                 num_input_channels,
                 conv_filter_size,
                 num_filters,
                 conv_strides = 1,
                 use_pool = True,
                 pool_ksize = 2,
                 pool_strides = 2 ):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, conv_strides, conv_strides, 1], padding='SAME')

    layer += biases
    layer = tf.nn.relu(layer)
    if use_pool:
        layer = tf.nn.max_pool(value=layer, ksize=[1, pool_ksize, pool_ksize, 1], strides=[1, pool_strides, pool_strides, 1], padding='SAME')

    return layer
def create_flatten_layer(layer):
    layer_shape=layer.get_shape()
    num_features=layer_shape[1:4].num_elements()
    layer=tf.reshape(layer,[-1,num_features])
    return layer
def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    dropout=True,
                    keep_prob=0.5,
                    use_relu=True):
    weights=create_weights(shape=[num_inputs,num_outputs])
    biases=create_biases(num_outputs)

    layer=tf.matmul(input,weights)+biases
    if dropout:
        layer=tf.nn.dropout(layer,keep_prob=keep_prob)
    if use_relu:
        layer=tf.nn.relu(layer)
    return layer


layer_conv1=create_convolution_layer(input=x,
                                     num_input_channels=num_channels,
                                     conv_filter_size=filter_size_conv1,
                                     num_filters=num_filters_conv1,
                                    conv_strides = conv_strides1,
                                     use_pool = use_pool1,
                                     pool_ksize = pool_ksize1,
                                     pool_strides = pool_strides1)
layer_conv2=create_convolution_layer(input=layer_conv1,
                                     num_input_channels=num_filters_conv1,
                                     conv_filter_size=filter_size_conv2,
                                     num_filters=num_filters_conv2,
                                    conv_strides = conv_strides2,
                                     use_pool = use_pool2,
                                     pool_ksize = pool_ksize2,
                                     pool_strides = pool_strides2)
layer_conv3=create_convolution_layer(input=layer_conv2,
                                     num_input_channels=num_filters_conv2,
                                     conv_filter_size=filter_size_conv3,
                                     num_filters=num_filters_conv3,
                                    conv_strides = conv_strides3,
                                     use_pool = use_pool3,
                                     pool_ksize = pool_ksize3,
                                     pool_strides = pool_strides3)
layer_conv4=create_convolution_layer(input=layer_conv3,
                                     num_input_channels=num_filters_conv3,
                                     conv_filter_size=filter_size_conv4,
                                     num_filters=num_filters_conv4,
                                    conv_strides = conv_strides4,
                                     use_pool = use_pool4,
                                     pool_ksize = pool_ksize4,
                                     pool_strides = pool_strides4)
layer_conv5=create_convolution_layer(input=layer_conv4,
                                     num_input_channels=num_filters_conv4,
                                     conv_filter_size=filter_size_conv5,
                                     num_filters=num_filters_conv5,
                                    conv_strides = conv_strides5,
                                     use_pool = use_pool5,
                                     pool_ksize = pool_ksize5,
                                     pool_strides = pool_strides5)
layer_conv6=create_convolution_layer(input=layer_conv5,
                                     num_input_channels=num_filters_conv5,
                                     conv_filter_size=filter_size_conv6,
                                     num_filters=num_filters_conv6,
                                    conv_strides = conv_strides6,
                                     use_pool = use_pool6,
                                     pool_ksize = pool_ksize6,
                                     pool_strides = pool_strides6)
layer_conv7=create_convolution_layer(input=layer_conv6,
                                     num_input_channels=num_filters_conv6,
                                     conv_filter_size=filter_size_conv7,
                                     num_filters=num_filters_conv7,
                                    conv_strides = conv_strides7,
                                     use_pool = use_pool7,
                                     pool_ksize = pool_ksize7,
                                     pool_strides = pool_strides7)
layer_conv8=create_convolution_layer(input=layer_conv7,
                                     num_input_channels=num_filters_conv7,
                                     conv_filter_size=filter_size_conv8,
                                     num_filters=num_filters_conv8,
                                    conv_strides = conv_strides8,
                                     use_pool = use_pool8,
                                     pool_ksize = pool_ksize8,
                                     pool_strides = pool_strides8)
layer_conv9=create_convolution_layer(input=layer_conv8,
                                     num_input_channels=num_filters_conv8,
                                     conv_filter_size=filter_size_conv9,
                                     num_filters=num_filters_conv9,
                                    conv_strides = conv_strides9,
                                     use_pool = use_pool9,
                                     pool_ksize = pool_ksize9,
                                     pool_strides = pool_strides9)

layer_flat=create_flatten_layer(layer_conv9)

layer_fc1=create_fc_layer(input=layer_flat,
                          num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                          num_outputs=fc_layer_size,
                          use_relu=True)
layer_fc2=create_fc_layer(input=layer_fc1,
                          num_inputs=fc_layer_size,
                          num_outputs=fc_layer_size2,
                          use_relu=True)
layer_fc3=create_fc_layer(input=layer_fc2,
                          num_inputs=fc_layer_size2,
                          num_outputs=num_classes,
                          dropout = False,
                          use_relu=False)
y_pred=tf.nn.softmax(layer_fc3,name='y_pred')
y_pred_cls=tf.argmax(y_pred,dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3,labels=y_true)
cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

session.run(tf.global_variables_initializer())

def show_progress(epoch,feed_dict_train,feed_dict_validate,acc_loss,val_loss,i):
    acc=session.run(accuracy,feed_dict=feed_dict_train)
    val_acc=session.run(accuracy,feed_dict=feed_dict_validate)
    print("epoch:",str(epoch+1)+",i:",str(i)+
          ",acc:",str(acc)+",acc_loss:",str(acc_loss)+",val_acc:",str(val_acc)+",val_loss:",str(val_loss))


total_iterations=0
saver=tf.train.Saver()

def train(num_iteration):
    global total_iterations
    for i in range(total_iterations,total_iterations+num_iteration):
        x_batch,y_true_batch,_,cls_batch=data.train.next_batch(batch_size)
        x_valid_batch,y_valid_batch,_,valid_cls_batch=data.valid.next_batch(batch_size)
        feed_dict_tr={x:x_batch,y_true:y_true_batch}
        feed_dict_val={x:x_valid_batch,y_true:y_valid_batch}

        session.run(optimizer,feed_dict=feed_dict_tr)
        examples=data.train.num_examples()
        #if i% int(examples/batch_size)==0:
        if i% int(examples)==0:
            val_loss=session.run(cost,feed_dict=feed_dict_val)
            acc_loss=session.run(cost,feed_dict=feed_dict_tr)
            #epoch=int(i/int(examples/batch_size))
            epoch=int(i/int(examples))

            show_progress(epoch,feed_dict_tr,feed_dict_val,acc_loss,val_loss,i)
            saver.save(session,'./model_25k/dog-cat.ckpt',global_step=i)
    total_iterations+=num_iteration


# In[2]:


train(num_iteration=120000)


# In[ ]:


sess=tf.Session()
sess.close()


# In[ ]:





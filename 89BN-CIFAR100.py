# coding:utf-8
#   对8-9cifar10分类程序进行改进解决100分类
import cifar_input
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm
import csv
import os
import sys
from six.moves import urllib
import tarfile
import conv_nets_test_2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_op_shape(t):
    '''
    输出一个操作op结点的形状
    '''
    print(t.op.name, '', t.get_shape().as_list())

#   第一步:数据准备
# 数据集中输入图像的参数
dataset_dir_cifar10 = '../CIFAR10_dataset/cifar-10-batches-bin'
dataset_dir_cifar100 = '../CIFAR100_dataset/cifar-100-binary'
dataset_dir_cifar10_root = '../CIFAR10_dataset'
dataset_dir_cifar100_root = '../CIFAR100_dataset'
num_examples_per_epoch_for_train = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN  # 50000
num_examples_per_epoch_for_eval = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL  # 10000
image_size = cifar_input.IMAGE_SIZE
image_channel = cifar_input.IMAGE_DEPTH

cifar10_data_url = cifar_input.CIFAR10_DATA_URL
cifar100_data_url = cifar_input.CIFAR100_DATA_URL
# 通过修改cifar10or20or100，就可以测试cifar10，cifar20，cifar100
# 或者使用假数据跑模型（让cifar10or20or100 = -1）
cifar10or20or100 = 100
if cifar10or20or100 == 10:
    n_classes = cifar_input.NUM_CLASSES_CIFAR10
    dataset_dir = dataset_dir_cifar10
    cifar_data_url = cifar10_data_url
    dataset_dir_cifar_root = dataset_dir_cifar10_root
if cifar10or20or100 == 20:
    n_classes = cifar_input.NUM_CLASSES_CIFAR20
    dataset_dir = dataset_dir_cifar100
    cifar_data_url = cifar100_data_url
    dataset_dir_cifar_root = dataset_dir_cifar100_root

if cifar10or20or100 == 100:
    n_classes = cifar_input.NUM_CLASSES_CIFAR100
    dataset_dir = dataset_dir_cifar100
    cifar_data_url = cifar100_data_url
    dataset_dir_cifar_root = dataset_dir_cifar100_root

def get_distorted_train_batch(data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    images, labels = cifar_input.distorted_inputs(cifar10or20or100=n_classes, data_dir=data_dir, batch_size=batch_size)
    return images, labels

def get_undistorted_eval_batch(data_dir, eval_data, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    images, labels = cifar_input.inputs(cifar10or20or100=n_classes, eval_data=eval_data, data_dir=data_dir,
                                        batch_size=batch_size)
    return images, labels

#   第一步:定义训练数据和测试数据
batch_size = 100
print ("begin to get train data and test data!")
#   训练数据
images_train, labels_train = get_distorted_train_batch(data_dir=dataset_dir, batch_size=batch_size)
#   测试数据
images_test, labels_test = get_undistorted_eval_batch(data_dir=dataset_dir, eval_data=True, batch_size=batch_size)
print ("begin data")
#   第二步:定义网络结构
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1],strides=[1, 6, 6, 1], padding='SAME')
#   在池化函数后面加入BN函数
def batch_norm_layer(value, train=None, name='batch_norm'):
    if train is not None:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=True)
    else:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=False)

#   为BN函数添加占位符参数
#   定义占位符
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
                        #   cifar data的shape 24*24*3
y = tf.placeholder(tf.float32, [None, 100])
                        #   0-9数字分类=> 10 classes
print_op_shape(y)
#   由于BN需要设置是否为训练状态,所以这里定义一个train将训练状态当成一个占位符来传入
train = tf.placeholder(tf.float32)
W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1, 32, 32, 3])    #32*32*3

#   修改网络结构添加BN层
#   在第一层h_conv1与第二层h_conv2 的输出之前卷积之后加入BN层
h_conv1 = tf.nn.relu(batch_norm_layer((conv2d(x_image, W_conv1) + b_conv1), train)) #32*32*64
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#   在第一层卷积层后添加一个的卷积层,使用封装好的库来构建,并对池化层也使用封装函数进行改写
h_conv1_2 = tf.contrib.layers.conv2d(h_conv1, 64, 5, 1, 'SAME', activation_fn=tf.nn.relu)
# h_pool1 = max_pool_2x2(h_conv1)
h_pool1 = tf.contrib.layers.max_pool2d(h_conv1_2, [2, 2], stride=2, padding='SAME') #16*16*64
print_op_shape(h_pool1)
#   将第二层卷积层拆成两个
W_conv2_1 = weight_variable([5, 1, 64, 128])
b_conv2_1 = bias_variable([128])
h_conv2_1 = tf.nn.relu(batch_norm_layer((conv2d(h_pool1, W_conv2_1) + b_conv2_1), train))
W_conv2 = weight_variable([1, 5, 128, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(batch_norm_layer((conv2d(h_conv2_1, W_conv2) + b_conv2), train))   #16*16*128
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
h_pool2 = tf.contrib.layers.max_pool2d(h_conv2, [2, 2], stride=2, padding='SAME')   #8*8*128
print_op_shape(h_pool2)
#   将第三层卷积层也拆成两个
W_conv3_1 = weight_variable([5, 1, 128, 256])
b_conv3_1 = bias_variable([256])
h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1)

W_conv3 = weight_variable([1, 5, 256, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_conv3_1, W_conv3) + b_conv3)  #8*8*256
print_op_shape(h_conv3)
#   在第三层卷积层后添加最大池化和两个全连接
h_pool3 = tf.contrib.layers.avg_pool2d(h_conv3, [2, 2], stride=2, padding='SAME')  # 池化4*4*256
print_op_shape(h_pool3)

#   全连接层1
nt_hpool3_1 = tf.contrib.layers.avg_pool2d(inputs=h_pool3, kernel_size=2, stride=2, padding='SAME') #2*2*256
print_op_shape(nt_hpool3_1)
nt_hpool3_1_flat = tf.reshape(nt_hpool3_1, [-1, 1024])  #1024*1
print_op_shape(nt_hpool3_1_flat)
y_conv_1 = tf.contrib.layers.fully_connected(inputs=nt_hpool3_1_flat, num_outputs=512, activation_fn=tf.nn.relu)
print_op_shape(y_conv_1)
y_conv_2 = tf.contrib.layers.fully_connected(inputs=y_conv_1, num_outputs=100, activation_fn=tf.nn.softmax)
print_op_shape(y_conv_2)
# nt_hpool3 = avg_pool_6x6(h_conv3)
# nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
# y_conv = tf.nn.softmax(nt_hpool3_flat)

#   加入退化学习率
#   将原来的学习率改成退化学习率,使用0.04的初值,让其每1000次退化0.9
#   cross_entroy = -tf.reduce_sum(y*tf.log(y_conv_2))
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv_2), axis=1))
#print ("cross_entroy:",cross_entroy)
global_step = tf.Variable(0, trainable=False)
decaylearning_rate = tf.train.exponential_decay(0.00001, global_step, 100, 0.9)
train_step = tf.train.AdamOptimizer(decaylearning_rate).minimize(cost,global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y, 1))
# print_op_shape(correct_prediction)
# print ("predicts_y",correct_prediction)
# print ("true_y", y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print ("accuracy", accuracy)
#   第三步:训练过程
#   启动session,迭代15000次数据集,这里要记着启动队列,同时读出来的label还要转成onehot编码
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
#   在Session中添加训练标志
#   在session中的循环部分,为占位符train添加数值1, 表明当前是训练状态.
#   因为前面BN函数中已经设定好train为None时,已经认为是测试状态.
for i in range(40001):  # 20000
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(100, dtype=float)[label_batch]       #   onehot编码]
    train_step.run(feed_dict={x: image_batch, y: label_b, train: 1}, session=sess)
    # print("label_b:", label_b)
    if i % 50 == 0:
        train_accyracy = accuracy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
        #print("label_b:", label_b)
        print ("step %d , training accuracy %g " % (i, train_accyracy))
    if i % 1000 == 0:
        #   每训练两遍进行一次测试.
        #   从测试集中将数据取出,放到模型中运行,查看模型的正确率
        print ("======>>>>every two times full-training,we will try once test.<<<<======")
        image_batch, label_batch = sess.run([images_test, labels_test])
        label_b = np.eye(100, dtype=float)[label_batch]
        print("finished! test accuracy %g" % accuracy.eval(feed_dict={x: image_batch, y: label_b}, session=sess))

#   第四步:评估结果



from keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
import tensorflow.keras.backend as K
#from tensorflow.keras.backend import tensorflow_backend
import tensorflow as tf
from tensorflow.keras import backend
from model.SEResNeXt import SEResNeXt
#from utils.img_util import arr_resize

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.preprocessing import label_binarize

from keras.callbacks import ModelCheckpoint
from scipy.io import loadmat
import scipy.io as scio

import json
import configparser

import numpy as np
import os
from PIL import Image,ImageEnhance,ImageChops
import matplotlib.pyplot as plt
import gc 
import pandas as pd
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
Height  = 300
height  = 300
Weight  = 256
width  = 256
batch_size = 128 
nb_epoch = 200
def brightnessEnhancement(image):#亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.1
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened
 

def move(image): #平移，平移尺度为off
    offset1 = ImageChops.offset(image, 100,150 )
    offset2 = ImageChops.offset(image, 100,-150 )
    offset3 = ImageChops.offset(image, -100,150 )
    offset4 = ImageChops.offset(image, -100,-150 )
    return offset1,offset2,offset3,offset4

def contrastEnhancement(image):#对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def load_Img0(imgDir,imgFoldName):
  imgs = os.listdir(imgDir+imgFoldName)
  imgNum = len(imgs)
  #imgNum = 10
  #print(imgNum)
  data = np.empty((imgNum,Weight,Height),dtype="float32")
  label = np.zeros((imgNum,13),dtype="uint8")
  for i in range (imgNum):
      img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
      gray=img.convert('L')#转换成灰度
      
#      offset1,offset2,offset3,offset4 = move(gray) #数据增强模块
#     
#      bright = brightnessEnhancement(cc)
#      contrast = contrastEnhancement(cc)
#      plt.imshow(contrast,cmap='gray' )
#      plt.axis('on')
#      plt.title('image')
#      plt.show()
      cc = gray.resize((Height,Weight))
#      plt.imshow(cc,cmap='gray' )
#      plt.axis('on')
#      plt.title('image')
#      plt.show()
      gray=np.array(cc, dtype='float32')
      #print(gray.shape)
      min1 = np.min(gray)
      max1 = np.max(gray)
      range1 =max1 - min1
      gray=(2*(gray-min1)/range1)-1
      arr=np.array(gray, dtype='float32')
      data[i,:,:] = arr
      #label[i] = int( 0 ) #int(imgs[i].split('.')[0])
  return data, label, imgNum


def load_Img1(imgDir,imgFoldName, CHD):
  imgs = os.listdir(imgDir+imgFoldName)
  df   = pd.read_excel(CHD, sheet_name='Sheet1',nrows=4287)
  data = df.values
  imgNum = len(imgs)
  #imgNum = 10
  temp = 0
  label = np.zeros((imgNum,13),dtype="uint8")
  for i in range(imgNum):
      for j in range(4287):
          label[i][12]=1
         
          if data[j,0] in imgs[i]:
              if 'L' in str(data[j,5]):
                   label[i][6]=1
              if 'X' in str(data[j,5]):
                   label[i][7]=1 
              if 'F' in str(data[j,5]):
                   label[i][8]=1 
              if 'Z' in str(data[j,5]):
                   label[i][9]=1 
              if 'D' in str(data[j,2]):
                   label[i][10]=1 
              if 'S' in str(data[j,2]):
                   label[i][11]=1      
              #print("num")
              #print(data[j,3])
              #no =  str(data[j,3]).split('，')
              no = re.findall(r'\d+', str(data[j,3]))
              lentf = len(no)
              for k in range(lentf):
                  aa =int( no[k] )
                  label[i][aa-1]=1
                  #print(aa)
              #print(no[])
              #print(data[j,0])
              #print(imgs[i])
              
              temp = temp+1
  #for i in range(imgNum):
      #print(label[i])
  #print(temp)            
              
  #print(imgNum)
  
  data = np.empty((imgNum,Weight,Height),dtype="float32")
 
  for i in range (imgNum):
      img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
      #print(imgs[i])
      gray=img.convert('L')#转换成灰度
      
      cc = gray.resize((Height,Weight))
      gray=np.array(cc, dtype='float32')
      #print(gray.shape)
      # gray.resize((300, 300, 1))  # 改变大小
      min1 = np.min(gray)
      max1 = np.max(gray)
      range1 =max1 - min1
      gray=(2*(gray-min1)/range1)-1
#      plt.imshow(gray,cmap='gray' )
#      plt.axis('on')
#      plt.title('image')
#      plt.show()
      arr=np.array(gray, dtype='float32')
      data[i,:,:] = arr
      #label[i] = int( 1 ) #int(imgs[i].split('.')[0])
  return data, label, imgNum



traincraterDir = "/home/star/Yu/senet-keras-master/gdph_dataset/training_set/"
trainfoldName0 = "0_normal_1"
trainfoldName1 = "1_CHD_1"

testcraterDir = "/home/star/Yu/senet-keras-master/gdph_dataset/test_set/"
testfoldName0 = "0_normal_1"
testfoldName1 = "1_CHD_1"


gongkaicraterDir = "/home/star/Yu/senet-keras-master/gongkai/"
gongkaifoldName1 = "NORMAL1"
gongkaifoldName11 = "NORMAL2"

validcraterDir = "/home/star/Yu/senet-keras-master/gdph_dataset/validation_set/"
validfoldName0 = "0_normal_1"
validfoldName1 = "1_CHD_1"

aa =  "/home/star/Yu/senet-keras-master/CHD统计1.xlsx"





# print('traindata')
# traindata0, trainlabel0, trainnum0 = load_Img0(traincraterDir,trainfoldName0)
# traindata1, trainlabel1, trainnum1 = load_Img1(traincraterDir,trainfoldName1,aa)
# print(trainnum0+trainnum1)
# print('validdata')
# validdata0, validlabel0, validnum0 = load_Img0(validcraterDir, validfoldName0)
# validdata1, validlabel1, validnum1 = load_Img1(validcraterDir, validfoldName1,aa)
# print(validnum0+validnum1)

# print('testdata')
# testdata0, testlabel0, testnum0 = load_Img0(testcraterDir,  testfoldName0)
# testdata1, testlabel1, testnum1 = load_Img1(testcraterDir,  testfoldName1,aa)
# print('totaltestdata')

# print(testnum0+testnum1)
# print('gongkai')
# gongkaidata1, gongkailabel1, gongkainum1 = load_Img0(gongkaicraterDir, gongkaifoldName1)
# gongkaidata11, gongkailabel11, gongkainum11 = load_Img0(gongkaicraterDir, gongkaifoldName11)
# print(gongkainum1+gongkainum11)

# print('totltrainnum:')
# traindata  = np.concatenate((traindata0,   traindata1,  validdata0,  validdata1,  testdata0,  testdata1,  gongkaidata1, gongkaidata11) , axis=0)
# trainlabel = np.concatenate(( trainlabel0, trainlabel1, validlabel0, validlabel1, testlabel0, testlabel1, gongkailabel1,gongkailabel11), axis=0)
# print(trainnum0+trainnum1+validnum0+validnum1+gongkainum1+gongkainum11+ testnum0+ testnum1)



# del traindata0, trainlabel0,traindata1, trainlabel1,validdata0, validlabel0,validdata1, validlabel1
# gc.collect()



# testdata   =  np.concatenate(( testdata0, testdata1),   axis=0)
# testlabel  =  np.concatenate(( testlabel0, testlabel1), axis=0) 


# ind = "/home/star/Yu/senet-keras-master/数据排序.xlsx"
# #index=np.arange(trainnum0+trainnum1+validnum0+validnum1+gongkainum1+gongkainum11+ testnum0+ testnum1)
# total =  trainnum0+trainnum1+validnum0+validnum1+gongkainum1+gongkainum11+ testnum0+ testnum1
# #np.random.shuffle(index)
# df   = pd.read_excel(ind, sheet_name='Sheet1',nrows=11988)
# data =  df.values
# index =data[:,8 ].astype(np.int64)
# print("********************************************")   
# print(index) 

  
# X_train=traindata[index,:,:]#X_train是训练集，y_train是训练标签
# X_train=np.reshape(X_train,((trainnum0+trainnum1+validnum0+validnum1+testnum0+ testnum1 + gongkainum1+gongkainum11),Height,Weight,1))
# Y_train=trainlabel[index]
# np.savetxt('indexhangzhour11'+'.txt',index )

# X_test=X_train[0:1000,:,:,:]
# Y_test=Y_train[0:1000]
# #X_test=np.reshape(X_test,((testnum0+testnum1),Height,Weight,1))

# X_train = X_train[1000:total,:,:,:]
# Y_train = Y_train[1000:total]
# temp = Y_train
# print('X_train.shape:')
# print(X_train.shape)
# #temp  = X_train[1000,:,:,:]
# #cus  = [0,2,0,3,2,3,4,3,0,4,0,3]
# cus  = [0,0,0,8,3,2,0,3,0,15,0,7]  
# #bb = [0.50, 0.70, 0.6,  0.7, 0.7, 0.7, 0.70, 0.7, 0.6, 0.7, 0.5,  0.65 ] 
# for i in range(12):
    # ins    = np.where(temp[:, i] > 0)
    # label  = Y_train[ins,:]
    # data   = X_train[ins,:,:,:]
    # data   = np.reshape(data,(data.shape[1],Height,Weight,1))
    # label  = np.reshape(label,(label.shape[1],13))
    # for j in range(cus[i]):
        # X_train   =  np.concatenate(( X_train, data),   axis=0)
        # Y_train   =  np.concatenate(( Y_train, label),   axis=0)

# index=np.arange(X_train.shape[0])
# np.random.shuffle(index)
# print('totaltrainsample')
# print(index)  
# X_train=X_train[index,:,:,:]#X_train是训练集，y_train是训练标签
# #X_train=np.reshape(X_train,((trainnum0+trainnum1+validnum0+validnum1+testnum0+ testnum1 + gongkainum1+gongkainum11),Height,Weight,1))
# Y_train=Y_train[index]   



# scio.savemat('NCTrain1.mat', {'A':X_train[0:7000,:,:,:],'B':Y_train[0:7000]}) 
# scio.savemat('NCTrain2.mat', {'A':X_train[7000:X_train.shape[0],:,:,:],'B':Y_train[7000:X_train.shape[0]]}) 
# scio.savemat('NCTest.mat', {'A':X_test,'B':Y_test})



data1 = loadmat('NCTrain1.mat')
data2 = loadmat('NCTrain2.mat')
data3 = loadmat('NCTest.mat')
            
            
train1    = data1['A']
label1 = data1['B']
            
            
train2    = data2['A']
label2 = data2['B']
            
X_test = data3['A']
Y_test = data3['B']
            
X_train = np.concatenate((train1,train2),axis=0)
Y_train = np.concatenate((label1,label2),axis=0)
#X_train是训练集，y_train是训练标签



# index=np.arange(testnum0+testnum1)
# np.random.shuffle(index)
  
# X_test=testdata[index,:,:]#X_train是训练集，y_train是训练标签
# Y_test=testlabel[index]
# X_test=np.reshape(X_test,((testnum0+testnum1),Height,Weight,1))


#X_train是训练集，y_train是训练标签

# session = tf.Session()
# K.set_session(session)

import tensorflow as tf


# __all__ = [
    # "hardest_triplet_loss", "semihard_triplet_loss"
# ]

"""
FaceNet: A Unified Embedding for Face Recognition and Clustering (CVPR "15)
Florian Schroff, Dmitry Kalenichenko, James Philbin
https://arxiv.org/abs/1503.03832
Implementation inspired by blogpost:
https://omoindrot.github.io/triplet-loss
triplet loss:
L = max(0, ||f(x_a) - f(x_p)||^2 - ||f(x_a) - f(x_n)||^2 + margin)
a = anchor
p = positive (label[a]==label[p])
n = negative (label[a]!=label[n])
easy triplets:
||f(x_a) - f(x_p)||^2 + margin < ||f(x_a) - f(x_n)||^2
positive is closer to anchor than negative by atleast margin
||f(x_a) - f(x_p)||^2 - ||f(x_a) - f(x_n)||^2 + margin = negative value
hard triplets:
||f(x_a) - f(x_n)||^2 < ||f(x_a) - f(x_p)||^2
negative is closer to anchor than positive
||f(x_a) - f(x_p)||^2 - ||f(x_a) - f(x_n)||^2 + margin = postive value
semi-hard triplets:
||f(x_a) - f(x_p)||^2 < ||f(x_a) - f(x_n)||^2 < ||f(x_a) - f(x_p)||^2 + margin
postivie is closer to anchor than negative but negative is within margin
||f(x_a) - f(x_p)||^2 - ||f(x_a) - f(x_n)||^2 + margin = postive value
"""


def pairwise_distances(embeddings):
    # get pariwise-distance matrix
    # p_dist_mat[i, j] = ||f(x_i) - f(x_j)||^2
    #                  = ||f(x_i)||^2  - 2 <f(x_i), f(x_j)> + ||f(x_j)||^2
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    p_dist_mat = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    p_dist_mat = tf.maximum(0.0, p_dist_mat)
    return p_dist_mat


def anchor_positive_mask(labels):
    diag = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    zero_diag = tf.logical_not(diag)
    labels_eq = tf.equal(labels, tf.transpose(labels))
    mask = tf.logical_and(labels_eq, zero_diag)
    return mask


def anchor_negative_mask(labels):
    labels_eq = tf.equal(labels, tf.transpose(labels))
    mask = tf.logical_not(labels_eq)
    return mask


def hardest_triplet_loss(labels, embeddings, margin=1.0):
    # get pairwise distances of embeddings
    pairwise_dist = pairwise_distances(embeddings)

    # get anchors to positives distances
    mask_ap = anchor_positive_mask(labels)
    mask_ap = tf.cast(mask_ap, tf.float32)

    # get hardest positive for each anchor
    ap_dist = tf.multiply(mask_ap, pairwise_dist)
    hardest_ap_dist = tf.reduce_max(ap_dist, axis=1, keepdims=True)

    # get anchors to negatives distances
    mask_an = anchor_negative_mask(labels)
    mask_an = tf.cast(mask_an, tf.float32)

    # add maximum distance to each positive (so we can get mininum negative distance)
    # get hardest negative for each anchor
    max_an_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    an_dist = pairwise_dist + max_an_dist * (1.0 - mask_an)
    hardest_an_dist = tf.reduce_min(an_dist, axis=1, keepdims=True)

    # calculate triplet loss using hardest positive and negatives of each anchor
    triplet_loss = tf.maximum(0.0, hardest_ap_dist - hardest_an_dist + margin)
    triplet_loss = tf.reduce_mean(triplet_loss)
    return triplet_loss



def similarity(labels, embeddings, margin=1.0):
    # get pairwise distances of embeddings
    pairwise_dist = pairwise_distances(embeddings)
    mask_ap = anchor_positive_mask(labels[:,0])
    mask_ap = tf.cast(mask_ap, tf.float32)
    for i in range(12):
        _ap = anchor_positive_mask(labels[:,i])
        _ap = tf.cast(_ap, tf.float32)
        mask_ap = tf.multiply(mask_ap,_ap)
    print('mask_ap.shape')    
    print(mask_ap.shape)
    # get hardest positive for each anchor
    ap_dist = tf.multiply(mask_ap, pairwise_dist)
    hardest_ap_dist = tf.reduce_max(ap_dist, axis=1, keepdims=True)

    # get anchors to negatives distances

    mask_an = anchor_negative_mask(labels[:,0])
    mask_an = tf.cast(mask_ap, tf.float32)
    for i in range(12):
        _an = anchor_negative_mask(labels[:,i])
        _an = tf.cast(_ap, tf.float32)
        mask_an = tf.multiply(mask_an,_an)
    # add maximum distance to each positive (so we can get mininum negative distance)
    # get hardest negative for each anchor
    max_an_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    an_dist = pairwise_dist + max_an_dist * (1.0 - mask_an)
    hardest_an_dist = tf.reduce_min(an_dist, axis=1, keepdims=True)

    # calculate triplet loss using hardest positive and negatives of each anchor
    triplet_loss = tf.maximum(0.0, hardest_ap_dist - hardest_an_dist + margin)
    triplet_loss = tf.reduce_mean(triplet_loss)
    return triplet_loss
    
    


def triplet_mask(labels):
    # get distinct indicies (i!=j, i!=k and j!=k)
    diag = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    zero_diag = tf.logical_not(diag)
    ij_not_eq = tf.expand_dims(zero_diag, 2)
    ik_not_eq = tf.expand_dims(zero_diag, 1)
    jk_not_eq = tf.expand_dims(zero_diag, 0)
    distinct_idxes = tf.logical_and(
        tf.logical_and(ij_not_eq, ik_not_eq), jk_not_eq)

    # get valid label indicies (label[i]==label[j] and label[i]!=label[k])
    label_eq = tf.equal(labels, tf.transpose(labels))
    ij_eq = tf.expand_dims(label_eq, 2)
    ik_eq = tf.expand_dims(label_eq, 1)
    ik_not_eq = tf.logical_not(ik_eq)
    valid_label_idxes = tf.logical_and(ij_eq, ik_not_eq)

    # combine distinct indicies and valid label indicies
    mask = tf.logical_and(distinct_idxes, valid_label_idxes)
    return mask


def semihard_triplet_loss(labels, embeddings, margin=1.0):
    # get pairwise distances of embeddings
    pairwise_dist = pairwise_distances(embeddings)

    # build 3d anchor/positive/negative distance tensor
    ap_dist = tf.expand_dims(pairwise_dist, 2)
    an_dist = tf.expand_dims(pairwise_dist, 1)
    triplet_loss = ap_dist - an_dist + margin

    # remove invalid triplets
    mask = triplet_mask(labels)
    mask = tf.cast(mask, tf.float32)
    triplet_loss = tf.multiply(triplet_loss, mask)

    # remove easy triplets (negative losses)
    triplet_loss = tf.maximum(0.0, triplet_loss)

    # get number of non-zero triplets and normalize
    pos_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_pos_triplets = tf.reduce_sum(pos_triplets)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_pos_triplets + 1e-16)
    return triplet_loss


def binary_focal_loss(y_true, y_pred, gamma=2, alpha=0.5):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    
    """
    y_true shape need be (None,1)
     y_pred need be compute after sigmoid
        """
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
    p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        
    return K.mean(focal_loss)

def knowledge_distillation_loss(y_true, y_pred):
    lambda_const = 1.0
    y_true1, y_true2 = y_true[:, :12], y_true[:, 12]
    
    y_pre, y_constrast, y_pre_feature = y_pred[:, :12], y_pred[:, 12], y_pred[:, 12:13+1600]
    
   
    aa = 0.0
    #bb = [0.70, 0.80, 0.7,  0.85, 0.85, 0.85, 0.70, 0.85, 0.75, 0.88, 0.5,  0.82 ] 
    bb = [0.50, 0.70, 0.6,  0.7, 0.7, 0.7, 0.70, 0.7, 0.6, 0.7, 0.5,  0.65 ] 
    w =  [1.00, 1.00, 1.0,  1.0,  1.0,  1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ] 
    for i in range(12):
        aa =aa + w[i]*binary_focal_loss(y_true1[:,i], y_pre[:,i], alpha=bb[i]) 
        #print(i)
    cc = 0.0    
    #cc = hardest_triplet_loss(y_true2, y_pre_feature )
    cc = hardest_triplet_loss(y_true2, y_pre_feature )
    
    # y1, y2, y3, y4, y5  = y_true[:, 0],  y_true[:, 1], y_true[:, 2], y_true[:, 3], y_true[:, 4],
    # where_index1 = tf.where(tf.equal(y1, 1))
    # data1  = tf.gather_nd(y_pre_feature, where_index1)
    # labels = tf.gather_nd(y1,where_index1)

    # dd = 0.0
    # dd = 0.5*hardest_triplet_loss(labels, data1) 
    
    dd = 0.0
    where_index1 = tf.where(tf.equal(y_true[:,12], 1))
    data1  = tf.gather_nd(y_pre_feature, where_index1)
    labels = tf.gather_nd(y_true1,where_index1)
    ee    =  similarity(labels, data1)
    
    for i in range(12):
        where_index1 = tf.where(tf.equal(y_true[:,i], 1))
        data1  = tf.gather_nd(y_pre_feature, where_index1)
        labels = tf.gather_nd(y_true[:,i],where_index1)
        if where_index1.shape[0]>1:
            dd =dd + bb[i]*similarity_loss(labels, data1)
        
    return  0.7*tf.keras.losses.binary_crossentropy(y_true2,  y_constrast)+ 1.0*tf.losses.mean_squared_error(y_true2,y_constrast)  + 0.7*aa + 0.0*cc + 0.0*dd + 0.0*ee

def acc(y_true, y_pred):
    y_true = y_true[:, 12]
    y_pred = y_pred[:, 12]
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)

def sen(y_true, y_pred):
    y_true = y_true[:, 12]
    y_pred = y_pred[:, 12]
    return tf.keras.metrics.TruePositives(y_true,  y_pred)

def detectionRate(y_true, y_pred):
    #true positive/all positive
    y_true = y_true[:, 12]
    y_pred = y_pred[:, 12]
    totalNum=tf.reduce_sum(y_true)
    tmp=tf.reduce_sum(tf.multiply(y_true, y_pred))
    return tf.reduce_sum(tmp)/totalNum
    
def lr_scheduler(epoch):
    if epoch % 30 == 0:
        K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * 0.1)
    return K.eval(model.optimizer.lr)
#change_lr = LearningRateScheduler(lr_scheduler)



# model.compile(
    # optimizer=sgd
    # , loss='binary_crossentropy'
    # , metrics=['binary_accuracy',tf.keras.metrics.SensitivityAtSpecificity(0.5, num_thresholds=1)])
#tf.metrics.true_positives()
 
from tensorflow.keras import layers
from tensorflow.keras import Model


filters = 64


channals = 1
nb_classes = 13

input_shape = (height, width, channals)

input_1 = layers.Input(shape=input_shape)

def BAM(input_tensor, r=8, d=4):

    input_dimension_shape = input_tensor.shape # (?, 28, 28, 64)

    _h = int(input_dimension_shape[1])
    _w = int(input_dimension_shape[2])
    num_channels = int(input_dimension_shape[3])

    # channel attention
    gap = layers.GlobalAveragePooling2D()(input_tensor) # (B, C)
    fc = layers.Dense(int(num_channels/r))(gap)
    c_attention = layers.Dense(num_channels)(fc)
    c_attention_bn = layers.BatchNormalization()(c_attention) # (B, C)


    # spatial attention
    conv_1_1 = layers.Conv2D(int(num_channels/r), 1, strides=1, padding="same", data_format='channels_last')(input_tensor)
    conv_3_3 = layers.Conv2D(int(num_channels/r), 3, strides=1, padding="same", dilation_rate=d, data_format='channels_last')(conv_1_1)
    conv_3_3 = layers.Conv2D(int(num_channels/r), 3, strides=1, padding="same", dilation_rate=d, data_format='channels_last')(conv_3_3)
    s_attention = layers.Conv2D(1, 1, strides=1, padding="same", data_format='channels_last')(conv_3_3)
    s_attention_bn = layers.BatchNormalization()(s_attention) # (B, H, W, 1)


    print("c_attention_bn", c_attention_bn)    # (?, 64)
    print("s_attention_bn", s_attention_bn)    # (?, 28, 28, 1)
    # projection
    c_att__w = layers.RepeatVector(_h*_w)(c_attention_bn) # (B, W, C) # (?, 28, 64) # (?, 784, 64)
    print("c_att__w", c_att__w)

    c_att__h_w = layers.Reshape([_h, _w, num_channels])(c_att__w)
    print("c_att__h_w", c_att__h_w)

    s_att__c = layers.Lambda(lambda x:backend.repeat_elements(x, num_channels, 3))(s_attention_bn) # (B, H, W, 1*num_channels)
    print("s_att__c", s_att__c)

    _sum = layers.Add()([c_att__h_w, s_att__c])
    bam = layers.Activation('sigmoid')(_sum)
    _mul = layers.Multiply()([input_tensor, bam])
    return _mul
    
    
def senet(inputs):
        reduction = 4
        channels = inputs.shape.as_list()[-1]
        avg_x = layers.GlobalAveragePooling2D()(inputs)
        avg_x = layers.Reshape((1,1,channels))(avg_x)
        avg_x = layers.Conv2D(int(channels)//reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu')(avg_x)
        avg_x = layers.Conv2D(int(channels),kernel_size=(1,1),strides=(1,1),padding='valid')(avg_x)
 
        max_x = layers.GlobalMaxPooling2D()(inputs)
        max_x = layers.Reshape((1,1,channels))(max_x)
        max_x = layers.Conv2D(int(channels)//reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu')(max_x)
        max_x = layers.Conv2D(int(channels),kernel_size=(1,1),strides=(1,1),padding='valid')(max_x)
 
        cbam_feature = layers.Add()([avg_x,max_x])
 
        cbam_feature = layers.Activation('hard_sigmoid')(cbam_feature)
 
        return layers.Multiply()([inputs,cbam_feature])


def conv1():    
    zpad = layers.ZeroPadding2D(padding=(3,3), data_format='channels_last')(input_1)
    '''
    加ZeroPadding2D是因为通过keras的API导出的ResNet_50模型中有ZeroPadding2D层，而ResNet_18和ResNet_50的开头都是一样的。
    '''
    con = layers.Conv2D(filters=16, kernel_size=(3,3),strides=(2,2), padding='valid')(zpad)
    bn = layers.BatchNormalization()(con)
    ac = layers.Activation('relu')(bn)
    zpad = layers.ZeroPadding2D()(ac)
    mp = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(zpad)
    return mp

##########################################################################
def conv2_x(cat):


    con = layers.Conv2D(filters=16, kernel_size=(3,3),strides=(1,1), padding='same')(cat)
    bn = layers.BatchNormalization()(con)
    ac = layers.Activation('relu')(bn)
    
    con = layers.Conv2D(filters=16, kernel_size=(3,3),strides=(1,1), padding='same')(ac)
    bn = layers.BatchNormalization()(con)
    
    conv2_x_add = layers.add([bn, cat])
   
    ac = layers.Activation('relu')(conv2_x_add)
    return ac

def conv3_x(cat, strides = 1):

    if strides == 2:
        con = layers.Conv2D(filters=16, kernel_size=(3,3),strides=(2,2), padding='same')(cat)#特征图尺寸减半，滤波器数量加倍。
    else:
        con = layers.Conv2D(filters=16, kernel_size=(3,3),strides=(1,1), padding='same')(cat)#相同的输出特征图，层具有相同数量的滤波器
    
    bn = layers.BatchNormalization()(con)
    ac = layers.Activation('relu')(bn)
    con = layers.Conv2D(filters=16, kernel_size=(3,3),strides=(1,1), padding='same')(ac)
    bn1 = layers.BatchNormalization()(con)

    if strides == 2:
        con = layers.Conv2D(filters=16, kernel_size=(3,3),strides=(2,2), padding='same')(cat)#特征图尺寸减半，滤波器数量加倍。
        bn2 = layers.BatchNormalization()(con)
        conv2_x_add = layers.add([bn1, bn2])
    else:
        conv2_x_add = layers.add([bn1, cat])#相同的输出特征图，层具有相同数量的滤波器
    
    ac = layers.Activation('relu')(conv2_x_add)
    return ac

def conv4_x(cat, strides = 1):

    if strides == 2:
        con = layers.Conv2D(filters=32, kernel_size=(3,3),strides=(2,2), padding='same')(cat)#特征图尺寸减半，滤波器数量加倍。
    else:
        con = layers.Conv2D(filters=32, kernel_size=(3,3),strides=(1,1), padding='same')(cat)#相同的输出特征图，层具有相同数量的滤波器
    
    bn = layers.BatchNormalization()(con)
    ac = layers.Activation('relu')(bn)
    con = layers.Conv2D(filters=32, kernel_size=(3,3),strides=(1,1), padding='same')(ac)
    bn1 = layers.BatchNormalization()(con)

    if strides == 2:
        con = layers.Conv2D(filters=32, kernel_size=(3,3),strides=(2,2), padding='same')(cat)#特征图尺寸减半，滤波器数量加倍。
        bn2 = layers.BatchNormalization()(con)
        conv2_x_add = layers.add([bn1, bn2])
    else:
        conv2_x_add = layers.add([bn1, cat])#相同的输出特征图，层具有相同数量的滤波器
    
    ac = layers.Activation('relu')(conv2_x_add)
    return ac


def conv5_x(cat, strides = 1):

    if strides == 2:
        con = layers.Conv2D(filters=64, kernel_size=(3,3),strides=(2,2), padding='same')(cat)#特征图尺寸减半，滤波器数量加倍。
    else:
        con = layers.Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), padding='same')(cat)#相同的输出特征图，层具有相同数量的滤波器
    
    bn = layers.BatchNormalization()(con)
    ac = layers.Activation('relu')(bn)
    con = layers.Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), padding='same')(ac)
    bn1 = layers.BatchNormalization()(con)

    if strides == 2:
        con = layers.Conv2D(filters=64, kernel_size=(3,3),strides=(2,2), padding='same')(cat)#特征图尺寸减半，滤波器数量加倍。
        bn2 = layers.BatchNormalization()(con)
        conv2_x_add = layers.add([bn1, bn2])
    else:
        conv2_x_add = layers.add([bn1, cat])#相同的输出特征图，层具有相同数量的滤波器
    
    ac = layers.Activation('relu')(conv2_x_add)
    return ac


##########################################################################
con1 = conv1()
##########################################################################
# con2 = conv2_x(con1)
# con2 = BAM(con2)
# con2 = conv2_x(con2)
con2 = senet(con1)
##########################################################################
# con3 = conv3_x(con2,strides=2)
# con3 = BAM(con3)
# con3 = conv3_x(con3,strides=1)
# con3 = senet(con3)
##########################################################################
con4 = conv4_x(con2, strides=2)
con4 = BAM(con4)
con4 = conv4_x(con4, strides=1)
con4 = senet(con4)
##########################################################################
con5 = conv5_x(con4, strides=2)
con5 = BAM(con5)
con5 = conv5_x(con5, strides=1)
con5 = senet(con5)
##########################################################################
avg = layers.AveragePooling2D(pool_size=(4,4), padding='same')(con5)
flatten = layers.Flatten()(avg)
if nb_classes == 1:
    dense = layers.Dense(units=nb_classes, activation='sigmoid')(flatten)
else:
    #dense1 = layers.Dense(units=256, activation='relu')(flatten)
    #dense1 = layers.Dropout(0.7)(dense1)
    #dense  = layers.Dense(units=nb_classes, activation='sigmoid')(dense1)
    dense  = layers.Dense(units=nb_classes, activation='sigmoid')(flatten)
    outputs = layers.concatenate([dense, flatten])

model = Model(inputs=[input_1], outputs=[outputs])
model.summary()

datagen = ImageDataGenerator(
    rescale = 1/255.
    # , shear_range = 0.1
    # , zoom_range = 0.1
    # , rotation_range=15
    # , width_shift_range=0.1
    # , height_shift_range=0.1
    #, featurewise_center = True
    )
    
# datagen = ImageDataGenerator(
    # rescale = 1/255.
    # )    
datagen.fit(X_train)

valid_datagen = ImageDataGenerator(rescale = 1/255.)
valid_datagen.fit(X_test)


model.compile(#loss='binary_crossentropy',
              loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred),
              #optimizer='rmsprop',
              #optimizer='adam',
              optimizer='sgd',
              #metrics=['binary_accuracy',tf.keras.metrics.SensitivityAtSpecificity(0.5, num_thresholds=1)]
              metrics = [acc,detectionRate]
              )
# model.fit(X_train, Y_train, batch_size=batch_size,
              # nb_epoch=nb_epoch,
              # validation_data = (X_test, Y_test),
              # #validation_split=0.1,
              # #callbacks=[change_lr],
              # shuffle=True)

checkpoint = ModelCheckpoint(filepath="best_weights.hdf5", monitor='val_acc',
verbose=1, save_best_only='False',  save_weights_only= 'False', mode='auto',period=1)
callback_lists = [checkpoint]   

model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=batch_size)
    , steps_per_epoch=len(X_train) // batch_size
    , epochs=400
    , validation_data = valid_datagen.flow(X_test, Y_test)
    , validation_steps=len(X_test) // batch_size
    , callbacks = callback_lists
    )



aa = model.predict(datagen.flow(X_test, Y_test))
y_pred = aa[:,12]
y_true = Y_test[:,12]

y_pred=np.array(y_pred,dtype=np.float32)

y_true=np.array(y_true,dtype=np.float32)

print(y_pred)
print(y_true)
fpr1,tpr1,threshold1 = roc_curve(y_true,y_pred , pos_label=1) ###计算真正率和假正率
roc_auc1 = auc(fpr1,tpr1) ###计算auc的值



model.load_weights('Zbest_weights.hdf5') 
#model2 = tf.keras.models.load_model('best_weights.hdf5', custom_objects={'knowledge_distillation_loss': knowledge_distillation_loss, 'acc': acc, 'detectionRate':detectionRate})
#model2 = tf.keras.models.load_model('best_weights.h5')
#model2.summary()
aa = model.predict(datagen.flow(X_test, Y_test))
y_pred = aa[:,12]
y_true = Y_test[:,12]

y_pred=np.array(y_pred,dtype=np.float32)

y_true=np.array(y_true,dtype=np.float32)

print(y_pred)
print(y_true)
fpr,tpr,threshold = roc_curve(y_true,y_pred , pos_label=1) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
 
plt.figure()
lw = 0.2
plt.figure(figsize=(7,7))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
         
         
plt.plot(fpr1, tpr1, color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('./testhangzhou.jpg')
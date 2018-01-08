#coding=utf-8
import os
import sys
import matplotlib.pyplot as plt

import caffe
import numpy as np
#pic_label=sys.argv[1]
#number=sys.argv[2]

def readImg(filelist):
    file_object = open(filelist)
    imgs=[]
    while True:
        line=file_object.readline()
        if not line:
            break
        imgs.append(line[0:len(line)-1])
    return imgs

def calcPrecision(tp,tn,fp,fn):
    return 1 if tp+fp==0 else float(tp)/(tp+fp)

def calcRecall(tp,tn,fp,fn):
    return 0 if tp+fn==0 else float(tp)/(tp+fn)

def plotPrecisionVsRecall(probs, answer):
    temp=sorted(np.insert(np.array(probs),2,np.array(answer),1).tolist())
    tp,tn,fp,fn=[0,0,0,0]
    precisions=[]
    recalls=[]
    tps=[]
    fps=[]
    #start from 0, treat all temp[][0]>0 as negative
    for k in range(len(temp)):
        if temp[k][2]==1:
            fn+=1
        else:
            tn+=1
    precisions.append(calcPrecision(tp,tn,fp,fn))
    recalls.append(calcRecall(tp,tn,fp,fn))
    tps.append(tp)
    fps.append(fp)
    for k in range(len(temp)):
        if temp[k][2]==1:
            tp+=1
            fn-=1
        else:
            fp+=1
            tn-=1
        tps.append(tp)
        fps.append(fp)
        precisions.append(calcPrecision(tp,tn,fp,fn))
        recalls.append(calcRecall(tp,tn,fp,fn))
    plt.figure(1)
    plt.plot(recalls,precisions, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0,1.1,0,1.1])
    plt.figure(2)
    plt.plot(np.array(fps)/np.double(tn+fp),np.array(tps)/np.double(tp+fn),linewidth=2)
    plt.axis([-0.1,1.1,0,1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()



if __name__ == "__main__":
    deploy='/home/capstone/Intel_face_detection/cnn/cnn_train_models/train_front/face_deploy.prototxt'    #deploy文件
    caffe_model='/home/capstone/Intel_face_detection/cnn/cnn_train_models/model_front/face_iter_1086450.caffemodel'   #训练好的 caffemodel
    posImgs=readImg('/home/capstone/Intel_face_detection_origin/adaboost/adaboost_test_origin/left_test/pos.txt')
    negImgs=readImg('/home/capstone/Intel_face_detection_origin/adaboost/adaboost_test_origin/left_test/neg.txt')
    mean_file='/home/capstone/Intel_face_detection/cnn/cnn_train_models/train_front/face_mean.npy'
    net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network
    #图片预处理设置
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
    transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
    transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR
    im=caffe.io.load_image(posImgs[0])                   #加载图片
    #net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中
    net.blobs['data'].reshape(len(posImgs)+len(negImgs),3,32,32)
    for k in range(len(posImgs)):
        net.blobs['data'].data[k,...] = transformer.preprocess('data',caffe.io.load_image(posImgs[k]))
    for k in range(len(negImgs)):
        net.blobs['data'].data[len(posImgs)+k,...] = transformer.preprocess('data',caffe.io.load_image(negImgs[k]))

#执行测试
    out = net.forward()
    probs=[]
    for k in range(len(net.blobs['prob'].data)):
        probs.append(net.blobs['prob'].data[k])  #取出最后一层（Softmax）属于某个类别的概率值，并打印
    plotPrecisionVsRecall(probs,[1]*len(posImgs)+[0]*len(negImgs))




#coding=utf-8
import os
import sys
import caffe
import numpy as np
#pic_label=sys.argv[1]
#number=sys.argv[2]

if __name__ == "__main__":
	deploy='/home/capstone/Intel_face_detection/cnn/cnn_train_models/train_front/face_deploy.prototxt'    #deploy文件
	caffe_model='/home/capstone/Intel_face_detection/cnn/cnn_train_models/model_front/face_iter_1086450.caffemodel'   #训练好的 caffemodel
	img='/home/capstone/Intel_face_detection_origin/adaboost/adaboost_train_origin/front_pos_sample/0/image00002_0.jpg' #随机找的一张待测图片
	mean_file='/home/capstone/Intel_face_detection/cnn/cnn_train_models/train_front/face_mean.npy'
	net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network
	#图片预处理设置
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
	transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
	transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
	transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
	transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR
	im=caffe.io.load_image(img)                   #加载图片
	net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中
#执行测试
	out = net.forward()

	prob= net.blobs['prob'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值，并打印
	print prob





 # Introduction
 Python script that can be used to classify input images using OpenCV and GoogLeNet (pre-trained on ImageNet) using the Caffe framework.
 
 The GoogLeNet architecture (now known as “Inception” after the novel micro-architecture) was introduced by Szegedy et al.
 in their 2014 paper, Going deeper with convolutions.
 
 # Usage
 
 python deep_learning_with_opencv.py --image images/jemma.png --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt

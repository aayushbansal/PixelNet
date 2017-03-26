# Training from Scratch

We randomly initialize the parameters of a VGG-16 network from a Gaussian distribution, and trained the models for semantic segmentation and surface normal estimation. 

1. Semantic Segmentation: We trained models for [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/index.html) data for 21 classes (20 categories + background). We augmented our dataset using the annotations provided by [Hariharan et al](http://home.bharathh.info/pubs/codes/SBD/download.html). The model was evaluated on the test set of PASCAL VOC-2012 using the evaluation server.

2. Surface Normal Estimation: We trained models for [surface normal estimation](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) using 795 unique trainval sequences (containing kinect data for each RGB frame). For details about dataset and how to download it, please refer to [MarrRevisited paper](https://github.com/aayushbansal/MarrRevisited).

Please consider citing these work if you happen to use these dataset.

## Notes
1. For semantic segmentation, we found that a linear model performs competitive to a MLP-based model while training a model from scratch. This is potentially due to small dataset size. 

2. The network tends to overfit when using conv-7 along with other layers (conv1_2, conv2_2, conv3_3, conv4_3, conv5_3)  in the hypercolumn feature vector. One way to avoid overfitting is to use dropout. Later, we also found that just using other five layers (without conv-7) also give competitive performance as the one with conv-7. 

3. For surface normal estimation, we trained the model  with the usual settings and did not observe the above mentioned things. We believe that such behaviour is observed due to small dataset size for semantic segmentation.

4. For object detection, please see Fast-RCNN [Fast RCNN](https://github.com/rbgirshick/fast-rcnn). Our models can be downloaded from here: 

```make

wget http://learn.perception.cs.cmu.edu/GitHub/PixelNet/scratch/det_via_normals.tar.gz 

```

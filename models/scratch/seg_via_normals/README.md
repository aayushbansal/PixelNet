# Self-supervised representation learning via geometry

The task of surface normal estimation does not require any human labels, and is primarily about capturing geometric information. We finetuned the model pre-trained for surface normal estimation (from scratch) for more semantic tasks: 

1. Semantic Segmentation: We finetuned model for [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/index.html) data for 21 classes (20 categories + background). We augmented our dataset using the annotations provided by [Hariharan et al](http://home.bharathh.info/pubs/codes/SBD/download.html). The model was evaluated on the test set of PASCAL VOC-2012 using the evaluation server.

2. Object Detection on PASCAL VOC-2007: We finetuned model for object detection on PASCAL VOC-2007 using [Fast RCNN](https://github.com/rbgirshick/fast-rcnn). We use the same settings as Fast RCNN except a step size of 70K, and fine-tuned the model for 200K iterations.

## Models
The model for semantic segmentation is same as the one described for scratch. Please see [Fast RCNN](https://github.com/rbgirshick/fast-rcnn) for object detection. 

The fine-tuned model can be downloaded using following: 

```make

wget http://learn.perception.cs.cmu.edu/GitHub/PixelNet/scratch/seg_via_normals.tar.gz ./
tar -xvzf seg_via_normals.tar.gz ./

```

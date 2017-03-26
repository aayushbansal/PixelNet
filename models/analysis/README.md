# Analysis

We consider a single-scale 224x224 input image and used two different tasks (a). semantic segmentation on [PASCAL VOC-2012](http://host.robots.ox.ac.uk/pascal/VOC/index.html); and (b). surface normal estimation on [NYU-depth dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). 

1. Semantic Segmentation: We trained models for [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/index.html) data for 21 classes (20 categories + background). We augmented our dataset using the annotations provided by [Hariharan et al](http://home.bharathh.info/pubs/codes/SBD/download.html). The model was evaluated on the test set of PASCAL VOC-2012 using the evaluation server.

2. Surface Normal Estimation: We trained models for [surface normal estimation](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) using 795 unique trainval sequences (containing kinect data for each RGB frame). For details about dataset and how to download it, please refer to [MarrRevisited paper](https://github.com/aayushbansal/MarrRevisited).

## Notes: 

1. The models used for Table-2 are contained in this folder. Refer to Section 4.2 (Linear vs. MLP) for this folder.

2. Note that we fine-tune the pretrained ImageNet model for this analysis, and we use hypercolumn features from conv-{1_2, 2_2, 3_3, 4_3, 5_3, 7} layers.

3. We found [batch-normalization](https://arxiv.org/abs/1502.03167) as an important tool along with our simple technique of _sampling_ to train linear models.

4. Without proper normalization of hypercolumn features, the training leads to a degenerate output for semantic segmentation.

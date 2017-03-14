# Models

Our [paper](http://www.cs.cmu.edu/~aayushb/pixelNet/pixelnet.pdf) have multiple analysis for different settings. Here we release the model description, and trained models for each of them. Our analysis are based on VGG-16 architecture.

## Section 1: Analysis
In this section, we consider a single-scale *224x224* input image and used two different tasks (a). semantic segmentation on [PASCAL VOC-2012](http://host.robots.ox.ac.uk/pascal/VOC/index.html); and (b). surface normal estimation on [NYU-depth dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). 

Here we show how to train linear and non-linear models effectively using skip connections. 

## Section 2: Scratch
We also trained our model from scratch for semantic segmentation and surface normal estimation. Since surface normal estimation (or as we call _Geometry_ in our paper) does not require any *human labels*, it can also be considered as a representation learnt in a  _self-supervised_ manner. For self-supervised representation learning, we evaluated our approach for semantic segmentation on PASCAL VOC-2012 and object detection on PASCAL VOC-2007.

We found [batch-normalization](https://arxiv.org/abs/1502.03167) as an important tool along with our simple technique of _sampling_ to train linear models, and training from scratch. 

## Section 3: Generalizability
Finally, we release the models for different tasks trained for different datasets: 

1. [PASCAL Context](http://www.cs.stanford.edu/~roozbeh/pascal-context/): We trained models for PASCAL Context dataset for 59 and 39 classes. The evaluation is done on the validation set. 

2. [PASCAL VOC-2012](http://host.robots.ox.ac.uk/pascal/VOC/index.html): We trained models for PASCAL VOC 2012 data for 21 classes (20 categories + background). We augmented our dataset using the annotations provided by [Hariharan et al](http://home.bharathh.info/pubs/codes/SBD/download.html). The model was evaluated on the test set of PASCAL VOC-2012 using the evaluation server.

3. [NYUv2 Surface Normal](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html): We trained models for surface normal estimation using 795 unique trainval sequences (containing kinect data for each RGB frame). For details about dataset and how to download it, please refer to [MarrRevisited paper](https://github.com/aayushbansal/MarrRevisited). 

4. [BSDS Edge Detection](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html): We trained models for edge detection on BSDS-500 dataset.

Please consider citing the above papers in case you use any of the above dataset or models trained using them.

## Contact
Please contact [Aayush Bansal](http://www.cs.cmu.edu/~aayushb/) if some model is missing or you want to know about some aditional analysis. 



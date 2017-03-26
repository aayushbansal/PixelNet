# Training a PixelNet model

Here we give an example of how to train a PixelNet model. We use task of semantic segmentation, and show how to train a model for PASCAL-VOC-2012 data using our scripts. We consider a single-scale 224x224 image here. 

## Data

Please run the following script to download the required data to train the model:

```make

sh download_data.sh

```

The script would do following things - 

1. Setup [PASCAL-VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/index.html).

2. Setup the augmented labels from [Hariharan et al.](http://home.bharathh.info/pubs/codes/SBD/download.html).

3. Setup the pre-trained [VGG-16 ImageNet model](https://arxiv.org/abs/1409.1556).

4. List of the images/segment labels used for training

You may skip using this script if you already have them on your machine.

## Training

Once you have setup all these things, you can train a model using 

```make

trainSeg(gpuID, options); % from the root-directory - mention gpuID

```

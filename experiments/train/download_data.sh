#!/bin/bash

echo "Setting up the required data"
mkdir "data";

# Setup data from PASCAL-VOC-2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar;
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar;
tar -xf VOCtrainval_11-May-2012.tar;
tar -xf VOCdevkit_18-May-2011.tar;
rm VOCtrainval_11-May-2012.tar;
rm VOCdevkit_18-May-2011.tar;
mv VOCdevkit data/;
cd "data/";
mkdir "PASCAL/"; cd "PASCAL/";
mkdir "VOC2012/"; cd "VOC2012/";
mv ../../VOCdevkit ./; 
cd ../../../;

# Setup data from Hariharan et al.
# Please consider citing them if you use this data
#wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz ./;
#tar -xvzf benchmark.tgz;
#mv benchmark_RELEASE bharath11;
#mv bharath11 data/;
#rm benchmark.tgz;
wget http://learn.perception.cs.cmu.edu/GitHub/PixelNet/bharath11.tar.gz;
tar -xvzf bharath11.tar.gz;
mv bharath11 data/;
rm bharath11.tar.gz;

# download the imagelist data from our server --
wget http://learn.perception.cs.cmu.edu/GitHub/PixelNet/train_data.tar.gz;
mv train_data.tar.gz data/;
cd "data/";
tar -xvzf train_data.tar.gz; 
rm train_data.tar.gz;
cd ../;

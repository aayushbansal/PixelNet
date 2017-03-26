# PixelNet: Representation _of_ the pixels, _by_ the pixels, and _for_ the pixels.

We explore design principles for general pixel-level prediction problems, from low-level edge detection to mid-level surface normal estimation to high-level semantic segmentation. Convolutional predictors, such as the fully-convolutional network (FCN), have achieved remarkable success by exploiting the spatial redundancy of neighboring pixels through convolutional processing. Though computationally efficient, we point out that such approaches are not statistically efficient during learning precisely because spatial redundancy limits the information learned from neighboring pixels. We demonstrate that stratified sampling of pixels allows one to:

1. add diversity during batch updates, speeding up learning; 

2. explore complex nonlinear predictors, improving accuracy; 

3. efficiently train state-of-the-art models *tabula rasa* (i.e., _from scratch_) for diverse pixel-labeling tasks. 

Our single architecture produces state-of-the-art results for semantic segmentation on PASCAL-Context dataset, surface normal estimation on NYUDv2 depth dataset, and edge detection on BSDS. We also demonstrate self-supervised representation learning via geometry. With even few data points, we achieve results better than previous approaches for unsupervised/self-supervised representation learning. More details are available on our [project page](http://www.cs.cmu.edu/~aayushb/pixelNet/). 

If you found these codes useful for your research, please consider citing -

```make
@article{pixelnet,
  title={PixelNet: {R}epresentation of the pixels, by the pixels, and for the pixels},
  author={Bansal, Aayush and Chen, Xinlei, and  Russell, Bryan and Gupta, Abhinav and Ramanan, Deva},
  Journal={arXiv preprint arXiv:1702.06506},
  year={2017}
}
```

## How to use these codes?

Anyone can freely use our codes for what-so-ever purpose they want to use. Here we give a detailed instruction to set them up and use for different applications. We will also provide the state-of-the-art models that we have trained. 

The codes can be downloaded using the following command:

```make 
git clone --recursive https://github.com/aayushbansal/PixelNet.git
cd PixelNet
```

Our codebase is built around [caffe](http://caffe.berkeleyvision.org/). We have included a pointer to caffe as a submodule. 

```make
ls tools/caffe
```

Our required layers are available within this submodule. To install Caffe, please follow the instructions on their [project page](http://caffe.berkeleyvision.org/).

## Models

We give a overview of the trained models in models/ directory.

```make
ls models
```

## Experiments

We provide scripts to train/test models in experiments/ directory.

```make
ls experiments
```

## Python Codes

The amazing [Xinlei Chen](https://www.cs.cmu.edu/~xinleic) wrote a version of code in [Python](https://github.com/endernewton/PixelNet) as well and wrote the GPU implementation of sampling layer. Check it out. We have not tested that extensively. It would be great if you could help us test that :)

# Demo 

## Example for Surface Normal Estimation
Here we give an example of testing a model for surface normal estimation. This code can be easily modified for other tasks. Note that this script is using a random surface normal model to demonstrate how to use the script.

Note: Please download the required data for the demo code using -

```make
wget http://learn.perception.cs.cmu.edu/GitHub/PixelNet/demo.tar.gz;
tar -xvzf demo.tar.gz;
rm demo.tar.gz;
```

## Example for Semantic Segmentation
Please use the script _demo-seg.m_, and download the required data for this demo - 

```make
wget http://learn.perception.cs.cmu.edu/GitHub/PixelNet/demo_seg.tar.gz;
tar -xvzf demo_seg.tar.gz;
rm demo_seg.tar.gz
```

## Example for Edge Detection
Please use the script for _demo-edges.m_, and download the required data for this demo - 

```make
wget http://learn.perception.cs.cmu.edu/GitHub/PixelNet/demo_edges.tar.gz;
tar -xvzf demo_edges.tar.gz;
rm demo_edges.tar.gz
```
### Note 
You may want to do the standard non-maximum suppression (NMS) and edge thinning for the outputs. In the demo code, we haven't done any of these, but it is required for evaluation on benchmarks. We followed [HED](https://github.com/s9xie/hed), and used Piotr Dollar's [Structured Forest matlab toolbox](https://github.com/pdollar/edges). Thank you Saining! Thank you Piotr!


#ifndef CAFFE_RAND_COMP_LAYER_HPP_
#define CAFFE_RAND_COMP_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Takes a Blob and crop it, to the shape specified by the second input
 *  Blob, across all dimensions after the specified axis.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
// added by ab --
// concat the points for different conv-layers
// randomly sample point from the images
// constraint -- all bottom blobs containing hypercol data
// should be consequential.
template <typename Dtype>
class RandCompLayer : public Layer<Dtype> {
 public:
  explicit RandCompLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RandComp"; }
  // should have at least 1 as input, and the sample as 2nd input
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  //shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 // no. of data points 
 int N_;
 // number of bottom blobs containing hypercol data 
 int n_hblobs_;
 // number of channels in the hypercol data --
 int n_channels_;
 // height, width
 vector<int> height_;
 vector<int> width_;
 vector<int> pixels_;
 // compression rate
 vector<int> comp_;
 // if do padding
 bool padded_;
 vector<Dtype> padding_;
 // store the pixels
 // Blob<Dtype> indexes_;
};


}  // namespace caffe

#endif  // CAFFE_RAND_COMP_LAYER_HPP_

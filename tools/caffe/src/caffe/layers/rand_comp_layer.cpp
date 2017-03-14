#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rand_comp_layer.hpp"

namespace caffe {

template <typename Dtype>
void RandCompLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // the no. of input layers
  n_hblobs_ = bottom.size() - 1;
  // get the last layer, which should be the indexes
  CHECK_EQ(bottom[n_hblobs_]->num_axes(), 2);
  CHECK_EQ(bottom[n_hblobs_]->shape(1), 3);
  N_ = bottom[n_hblobs_]->shape(0);
  CHECK_GT(N_,0);
  comp_ = vector<int>(n_hblobs_);
  padding_ = vector<Dtype>(n_hblobs_);
  RandCompParameter param = this->layer_param_.rand_comp_param();
  CHECK_EQ(param.compression_rate_size(), n_hblobs_);
  padded_ = param.pad();
  for (int i = 0; i < n_hblobs_; i++) {
    comp_[i] = param.compression_rate(i);
    if (padded_)
      padding_[i] = static_cast<Dtype>((comp_[i] - 1.0)/2);
    else
      padding_[i] = 0.0;
  }
}

template <typename Dtype>
void RandCompLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // also push in the size
  n_channels_ = 0;
  height_ = vector<int>(n_hblobs_);
  width_ = vector<int>(n_hblobs_);
  pixels_ = vector<int>(n_hblobs_);
  for (int i = 0; i < n_hblobs_; i++) {
    n_channels_ += bottom[i]->channels();
    height_[i] = bottom[i]->height();
    width_[i] = bottom[i]->width();
    pixels_[i] = height_[i] * width_[i];
  }
  N_ = bottom[n_hblobs_]->num();
  CHECK_GT(N_,0);
  vector<int> top_shape(2);
  top_shape[0] = N_;
  top_shape[1] = n_channels_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RandCompLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[n_hblobs_]->cpu_data();
  vector<const Dtype*> bottom_layers(n_hblobs_);
  for (int i = 0; i < n_hblobs_; i++) {
    bottom_layers[i] = bottom[i]->cpu_data();
  }

  int i = 0;
  int j = 0;
  int n, ch, tx, ty;
  Dtype x, y;
  int s = 0; // index for the output
  // const Dtype* bottom_layers;
  for (; i < N_; i++) {
    n = static_cast<int>(bottom_data[j++]);
    CHECK_GE(n, 0);
    y = bottom_data[j++];
    x = bottom_data[j++];
    // then find the corresponding locations
    for (int b = 0; b < n_hblobs_; b++) {
      // bottom_layers = bottom[b]->cpu_data();
      tx = static_cast<int>(round((x-padding_[b])/comp_[b]));
      ty = static_cast<int>(round((y-padding_[b])/comp_[b]));
      // check if they are within the size limit
      CHECK_GE(tx, 0);
      CHECK_LT(tx, width_[b]);
      CHECK_GE(ty, 0);
      CHECK_LT(ty, height_[b]);
      CHECK_LT(n, bottom[b]->num());
      ch = bottom[b]->channels();
      int init = n * ch * pixels_[b] + ty * width_[b] + tx;
      top_data[s++] = bottom_layers[b][init];
      for (int c = 1; c < ch; c++) {
        init += pixels_[b];
        top_data[s++] = bottom_layers[b][init];
      }
    }
  }
}

template <typename Dtype>
void RandCompLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[n_hblobs_]->cpu_data();
    vector<Dtype*> bottom_layers(n_hblobs_);
    int offset = 0;
    for (int b = 0; b < n_hblobs_; b++) {
      bottom_layers[b] = bottom[b]->mutable_cpu_diff();
      caffe_set(bottom[b]->count(), Dtype(0.), bottom_layers[b]);

      // then just get all the things for that layer, FUCK
      int i = 0;
      int j = 0;
      int n, tx, ty;
      Dtype x, y;
      int ch = bottom[b]->channels();
      int s = offset;
      int t;
      for (; i < N_; i++) {
        n = static_cast<int>(bottom_data[j++]);
        CHECK_GE(n, 0);
        y = bottom_data[j++];
        x = bottom_data[j++];
        // then find the corresponding locations
        tx = static_cast<int>(round((x-padding_[b])/comp_[b]));
        ty = static_cast<int>(round((y-padding_[b])/comp_[b]));
        CHECK_GE(tx, 0);
        CHECK_LT(tx, width_[b]);
        CHECK_GE(ty, 0);
        CHECK_LT(ty, height_[b]);
        CHECK_LT(n, bottom[b]->num());
        t = s;
        int init = n * ch * pixels_[b] + ty * width_[b] + tx;
        bottom_layers[b][init] += top_diff[t++];
        for (int c = 1; c < ch; c++) {
          init += pixels_[b];
          bottom_layers[b][init] += top_diff[t++];
        }
        s += n_channels_;
      }
      offset += ch;
    }

    // int i = 0;
    // int j = 0;
    // int n, ch, tx, ty;
    // Dtype x, y;
    // int s = 0; // index for the output
    // // Dtype* bottom_layers;
    // for (; i < N_; i++) {
    //   n = static_cast<int>(bottom_data[j++]);
    //   CHECK_GE(n, 0);
    //   y = bottom_data[j++];
    //   x = bottom_data[j++];
    //   // LOG(INFO) << n << " " << y << " " << x;
    //   // then find the corresponding locations
    //   for (int b = 0; b < n_hblobs_; b++) {
    //     // bottom_layers = bottom[b]->mutable_cpu_diff();
    //     tx = static_cast<int>(round((x-padding_[b])/comp_[b]));
    //     ty = static_cast<int>(round((y-padding_[b])/comp_[b]));
    //     // Dtype scomp = comp_[b] * comp_[b];
    //     CHECK_GE(tx, 0);
    //     CHECK_LT(tx, width_[b]);
    //     CHECK_GE(ty, 0);
    //     CHECK_LT(ty, height_[b]);
    //     CHECK_LT(n, bottom[b]->num());
    //     ch = bottom[b]->channels();
    //     int init = n * ch * pixels_[b] + ty * width_[b] + tx;
    //     LOG(INFO) << n << " " << ch << " " << pixels_[b] << " " << ty << " " << width_[b] << " " << tx;
    //     bottom_layers[b][init] += top_diff[s++];
    //     for (int c = 1; c < ch; c++) {
    //       init += pixels_[b];
    //       bottom_layers[b][init] += top_diff[s++];
    //     }
    //   }
    // }

  }
}

// #ifdef CPU_ONLY
// STUB_GPU(RandCompLayer);
// #endif

INSTANTIATE_CLASS(RandCompLayer);
REGISTER_LAYER_CLASS(RandComp);

}  // namespace caffe

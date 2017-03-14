#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rand_bi_layer.hpp"

namespace caffe {

template <typename Dtype>
void RandBILayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // the no. of input layers
  n_hblobs_ = bottom.size() - 1;
  // get the last layer, which should be the indexes
  CHECK_EQ(bottom[n_hblobs_]->num_axes(), 2);
  CHECK_EQ(bottom[n_hblobs_]->shape(1), 3);
  N_ = bottom[n_hblobs_]->shape(0);
  CHECK_GT(N_,0);
  // indexes_.Reshape(bottom[n_hblobs_]->shape());
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
void RandBILayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
  // indexes_.Reshape(bottom[n_hblobs_]->shape());
  vector<int> top_shape(2);
  top_shape[0] = N_;
  top_shape[1] = n_channels_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RandBILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[n_hblobs_]->cpu_data();
  // caffe_copy(bottom[n_hblobs_]->count(), bottom[n_hblobs_]->cpu_data(), indexes_.mutable_cpu_data());
  // const Dtype* bottom_data = indexes_.cpu_data();
  vector<const Dtype*> bottom_layers(n_hblobs_);
  for (int i = 0; i < n_hblobs_; i++) {
    bottom_layers[i] = bottom[i]->cpu_data();
  }

  int i = 0;
  int j = 0;
  int n, ch, tx1, tx2, ty1, ty2;
  Dtype tx, ty, rx, ry;
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
      tx = (x-padding_[b])/comp_[b];
      ty = (y-padding_[b])/comp_[b];
      tx1 = static_cast<int>(floor(tx));
      ty1 = static_cast<int>(floor(ty));
      tx2 = static_cast<int>(ceil(tx));
      ty2 = static_cast<int>(ceil(ty));
      // check if they are within the size limit
      // CHECK_GE(tx1, 0);
      tx1 = tx1 > 0 ? tx1 : 0;
      tx2 = tx2 > 0 ? tx2 : 0;

      CHECK_LT(tx2, width_[b]);
      // CHECK_GE(ty1, 0);
      ty1 = ty1 > 0 ? ty1 : 0;
      ty2 = ty2 > 0 ? ty2 : 0;

      CHECK_LT(ty2, height_[b]);
      CHECK_LT(n, bottom[b]->num());
      ch = bottom[b]->channels();
      // just check different cases for this thing..
      if ((tx1 == tx2) && (ty1 == ty2)) {
        int init = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
        top_data[s++] = bottom_layers[b][init];
        for (int c = 1; c < ch; c++) {
          init += pixels_[b];
          top_data[s++] = bottom_layers[b][init];
        }
      } else if (ty1 == ty2) {
        rx = tx - tx1;
        int init1 = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
        int init2 = init1 + 1;
        top_data[s++] = bottom_layers[b][init1] * (1.-rx) + bottom_layers[b][init2] * rx;
        for (int c = 1; c < ch; c++) {
          init1 += pixels_[b];
          init2 += pixels_[b];
          top_data[s++] = bottom_layers[b][init1] * (1.-rx) + bottom_layers[b][init2] * rx;
        }
      } else if (tx1 == tx2) {
        ry = ty - ty1;
        int init1 = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
        int init2 = init1 + width_[b];
        top_data[s++] = bottom_layers[b][init1] * (1.-ry) + bottom_layers[b][init2] * ry;
        for (int c = 1; c < ch; c++) {
          init1 += pixels_[b];
          init2 += pixels_[b];
          top_data[s++] = bottom_layers[b][init1] * (1.-ry) + bottom_layers[b][init2] * ry;
        }
      } else {
        rx = tx - tx1;
        ry = ty - ty1;
        int init11 = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
        int init12 = init11 + 1;
        int init21 = init11 + width_[b];
        int init22 = init21 + 1;
        top_data[s++] = (bottom_layers[b][init11] * (1.-ry) + bottom_layers[b][init21] * ry) * (1.-rx) +
              (bottom_layers[b][init12] * (1.-ry) + bottom_layers[b][init22] * ry) * rx;
        for (int c = 1; c < ch; c++) {
          init11 += pixels_[b];
          init12 += pixels_[b];
          init21 += pixels_[b];
          init22 += pixels_[b];
          top_data[s++] = (bottom_layers[b][init11] * (1.-ry) + bottom_layers[b][init21] * ry) * (1.-rx) +
                (bottom_layers[b][init12] * (1.-ry) + bottom_layers[b][init22] * ry) * rx;
        }
      }
    }
  }
}

template <typename Dtype>
void RandBILayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[n_hblobs_]->cpu_data();
    // const Dtype* bottom_data = indexes_.cpu_data();
    vector<Dtype*> bottom_layers(n_hblobs_);
    for (int i = 0; i < n_hblobs_; i++) {
      bottom_layers[i] = bottom[i]->mutable_cpu_diff();
      caffe_set(bottom[i]->count(), Dtype(0.), bottom_layers[i]);
      // caffe_memset(bottom[i]->count(), 0, bottom[i]->mutable_cpu_diff());
    }
    int i = 0;
    int j = 0;
    int n, ch, tx1, tx2, ty1, ty2;
    Dtype tx, ty, rx, ry;
    Dtype x, y;
    int s = 0; // index for the output
    // Dtype* bottom_layers;
    for (; i < N_; i++) {
      n = static_cast<int>(bottom_data[j++]);
      CHECK_GE(n, 0);
      y = bottom_data[j++];
      x = bottom_data[j++];
      // then find the corresponding locations
      for (int b = 0; b < n_hblobs_; b++) {
        // bottom_layers = bottom[b]->mutable_cpu_diff();
        tx = (x-padding_[b])/comp_[b];
        ty = (y-padding_[b])/comp_[b];
        // Dtype scomp = comp_[b] * comp_[b];

        tx1 = static_cast<int>(floor(tx));
        ty1 = static_cast<int>(floor(ty));
        tx2 = static_cast<int>(ceil(tx));
        ty2 = static_cast<int>(ceil(ty));
        // check if they are within the size limit
        // CHECK_GE(tx1, 0);
        tx1 = tx1 > 0 ? tx1 : 0;
        tx2 = tx2 > 0 ? tx2 : 0;

        CHECK_LT(tx2, width_[b]);
        // CHECK_GE(ty1, 0);
        ty1 = ty1 > 0 ? ty1 : 0;
        ty2 = ty2 > 0 ? ty2 : 0;

        CHECK_LT(ty2, height_[b]);
        CHECK_LT(n, bottom[b]->num());
        ch = bottom[b]->channels();
        // just check different cases for this thing..
        if ((tx1 == tx2) && (ty1 == ty2)) {
          int init = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
          bottom_layers[b][init] += top_diff[s++];
          for (int c = 1; c < ch; c++) {
            init += pixels_[b];
            bottom_layers[b][init] += top_diff[s++];
          }
        } else if (ty1 == ty2) {
          rx = tx - tx1;
          int init1 = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
          int init2 = init1 + 1;
          bottom_layers[b][init1] += top_diff[s] * (1.-rx);
          bottom_layers[b][init2] += top_diff[s++] * rx;
          for (int c = 1; c < ch; c++) {
            init1 += pixels_[b];
            init2 += pixels_[b];
            bottom_layers[b][init1] += top_diff[s] * (1.-rx);
            bottom_layers[b][init2] += top_diff[s++] * rx;
          }
        } else if (tx1 == tx2) {
          ry = ty - ty1;
          int init1 = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
          int init2 = init1 + width_[b];
          bottom_layers[b][init1] += top_diff[s] * (1.-ry);
          bottom_layers[b][init2] += top_diff[s++] * ry;
          for (int c = 1; c < ch; c++) {
            init1 += pixels_[b];
            init2 += pixels_[b];
            bottom_layers[b][init1] += top_diff[s] * (1.-ry);
            bottom_layers[b][init2] += top_diff[s++] * ry;
          }
        } else {
          rx = tx - tx1;
          ry = ty - ty1;
          int init11 = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
          int init12 = init11 + 1;
          int init21 = init11 + width_[b];
          int init22 = init21 + 1;
          bottom_layers[b][init11] += top_diff[s] * (1.-ry) * (1.-rx);
          bottom_layers[b][init21] += top_diff[s] * ry * (1.-rx);
          bottom_layers[b][init12] += top_diff[s] * (1.-ry) * rx;
          bottom_layers[b][init22] += top_diff[s++] * ry * rx;
          for (int c = 1; c < ch; c++) {
            init11 += pixels_[b];
            init12 += pixels_[b];
            init21 += pixels_[b];
            init22 += pixels_[b];
            bottom_layers[b][init11] += top_diff[s] * (1.-ry) * (1.-rx);
            bottom_layers[b][init21] += top_diff[s] * ry * (1.-rx);
            bottom_layers[b][init12] += top_diff[s] * (1.-ry) * rx;
            bottom_layers[b][init22] += top_diff[s++] * ry * rx;
          }
        }
      }
    }
  }
}

// #ifdef CPU_ONLY
// STUB_GPU(RandBILayer);
// #endif

INSTANTIATE_CLASS(RandBILayer);
REGISTER_LAYER_CLASS(RandBI);

}  // namespace caffe

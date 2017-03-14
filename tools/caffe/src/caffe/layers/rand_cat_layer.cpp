#include <vector>
#include <math.h>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rand_cat_layer.hpp"
#include <iostream>
//#include "caffe/util/rng.hpp"
#include <ctime>

namespace caffe {

template <typename Dtype>
void RandCatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // if random sampling is required else choose all the points --
  if_rand_ = this->layer_param_.rand_cat_param().rand_selection();

  // hard-coding start position and end position
  start_id_ = 0;
  n_hblobs_ = bottom.size() - 2;
  end_id_ = n_hblobs_ - 1;

  // assign the no. of points to be used --
  const int bottom_height = bottom[start_id_]->height();
  const int bottom_width = bottom[start_id_]->width();
  
  if(if_rand_){
 	  N_ = this->layer_param_.rand_cat_param().num_output(); 
   } 
   else {
	  N_ = bottom_height*bottom_width;
   } 

}

template <typename Dtype>
void RandCatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // set the top-layer to be nxc -- 
  vector<int> top_shape(2);
  top_shape[0] = N_*(bottom[start_id_]->num());

  // compute num-channels for the given bottom-data
  n_channels_ = 0;
  for (int i = start_id_; i <= end_id_; ++i) {
	n_channels_ = n_channels_ + bottom[i]->channels();
  }
  //LOG(INFO) << "NUM Channels: " << n_channels_;
  top_shape[1] = n_channels_;
  top[0]->Reshape(top_shape);

  // set the surface-norm layer to be nx3 --
  top_shape[1] = 3; // change the no. of channels 
  top[1]->Reshape(top_shape);

}

template <typename Dtype>
void RandCatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   
  // At training time, randomly select N_ points per image
  // At test time, take all the points in the image (1 image at test time)
  const Dtype* valid_data = bottom[end_id_+1]->cpu_data();
  rand_points_.clear();
  if(if_rand_) {
          // generate the list of points --
          std::srand ( unsigned ( std::time(0) ) );
          std::vector<int> shuffle_data_points;
          const int num_data_points = (bottom[start_id_]->height())*(bottom[start_id_]->width()); 
          for(int i = 0; i < num_data_points; i++){
              shuffle_data_points.push_back(i);
          }

	  for(int i = 0; i < (bottom[start_id_]->num()); i++) {
		// shuffle the points in the image --	
		std::random_shuffle(shuffle_data_points.begin(), shuffle_data_points.end());
		// find the N-valid-points from a image --
		int cnt_vp = 0;
		for(int j = 0; j < num_data_points; j++){
			int j_pt = shuffle_data_points[j];
			int data_pt = valid_data[j_pt];
			if(data_pt == 1) {
				cnt_vp++;
				rand_points_.push_back(i);
				rand_points_.push_back(j_pt);
			}
			if(cnt_vp >= N_){
				break;
			}
		}
	  }
	  shuffle_data_points.clear(); 
  } else {
	  // considering all the data points are considered --
	  for (int i = 0; i < (bottom[start_id_]->num()); i++) {
		for (int j = 0; j < N_; j++) {
			rand_points_.push_back(i);
			rand_points_.push_back(j);
		}        	     
           }
  }
 
  // TODO - can use memset to initialize to zero --
  Dtype* top_data = top[0]->mutable_cpu_data();
  for(int i = 0; i < top[0]->count(); i++) {
                top_data[i] = 0;
  }
  // TODO - can use memset to intialize to zero --
  Dtype* top_sn = top[1]->mutable_cpu_data();
  for(int i = 0; i < top[1]->count(); i++) {
               top_sn[i] = 0;
  }
  
  //
  const int bottom_width = bottom[start_id_]->width();
  const int bottom_height = bottom[start_id_]->height();
  const int bottom_nums = bottom[start_id_]->num();

  // get the data --
  std::vector<const Dtype*> bottom_layers(n_hblobs_);
  for (int i = 0; i < n_hblobs_; i++) {
    bottom_layers[i] = bottom[i]->cpu_data();
  }
  const Dtype* sn_data = bottom[end_id_+2]->cpu_data();

  // get the hypercolumn features for the selected points --
  int i = 0; int j = 0;
  int n, xy_l;
  int s = 0; int s_sn = 0;
  for(; i < N_*bottom_nums; i++) {
	n = int(rand_points_[j++]);
	xy_l = int(rand_points_[j++]);
        // then find the corresponding locations
   	for (int b = 0; b < n_hblobs_; b++) {
      		int init = n *(bottom[b]->channels())*bottom_width*bottom_height + xy_l;
      		top_data[s++] = bottom_layers[b][init];
      		for (int c = 1; c < bottom[b]->channels(); c++) {
        		init += bottom_width*bottom_height;
        		top_data[s++] = bottom_layers[b][init];
     		 }
    	}
	// and accumulate the surface normals (hard-coded for surface normals)
	for(int bc = 0; bc < 3; bc++){
		int init_sn = n*3*bottom_width*bottom_height + 
			      bc*bottom_width*bottom_width + xy_l;
		top_sn[s_sn++] = sn_data[init_sn];
	}
  }
}

template <typename Dtype>
void RandCatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 
    if(propagate_down[0]) {
	   //
  	  const int bottom_width = bottom[start_id_]->width();
 	  const int bottom_height = bottom[start_id_]->height();
  	  const int bottom_nums = bottom[start_id_]->num();
	 
	   //
	  const Dtype* top_diff = top[0]->cpu_diff();
	  std::vector<Dtype*> bottom_layers(n_hblobs_);
  	  for (int i = 0; i < n_hblobs_; i++) {
    		bottom_layers[i] = bottom[i]->mutable_cpu_diff();
		for(int j = 0; j < bottom[i]->count(); j++){
			bottom_layers[i][j] = 0;
		}
 	  }
	
	  // back-propagate to the layers -- 
	  int i = 0; int j = 0;
 	  int n, xy_l;
  	  int s = 0;
  	   for(; i < N_*bottom_nums; i++) {
        	n = int(rand_points_[j++]);
        	xy_l = int(rand_points_[j++]);
        	// then find the corresponding locations
        	for (int b = 0; b < n_hblobs_; b++) {
                	int init = n * (bottom[b]->channels())*bottom_width*bottom_height + xy_l;
			bottom_layers[b][init] = top_diff[s++];
                	for (int c = 1; c < bottom[b]->channels(); c++) {
                        	init += bottom_width*bottom_height;
                        	bottom_layers[b][init] = top_diff[s++];
                 	}
        	}
	  }
      }
}

INSTANTIATE_CLASS(RandCatLayer);
REGISTER_LAYER_CLASS(RandCat);

}  // namespace caffe

#include <vector>
#include <math.h>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rand_cat_conv_layer.hpp"
#include <iostream>
//#include "caffe/util/rng.hpp"
#include <ctime>

namespace caffe {

template <typename Dtype>
void RandCatConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  RandCatConvParameter params_ = this->layer_param_.rand_cat_conv_param();
  // hard-coded start position and end position
  start_id_ = 0;
  n_hblobs_ = bottom.size() - 2;
  end_id_ = n_hblobs_ - 1;

  // if random sampling is required else choose all the points --
  if_rand_ = params_.rand_selection();  
  if(if_rand_){
 	  N_ = params_.num_output(); 
  } else {
	  N_ = (bottom[start_id_]->height() - 2*params_.pad_factor())*
	       (bottom[start_id_]->width() - 2*params_.pad_factor());
  } 

  // get the pooling factor for the conv-layers
  // and their corresponding padding requirements --
  poolf_ = std::vector<int>(n_hblobs_);
  padf_ = std::vector<Dtype>(n_hblobs_);
  CHECK_EQ(params_.pooling_factor_size(), n_hblobs_);
  for (int i = 0; i < n_hblobs_; i++) {
    poolf_[i] = params_.pooling_factor(i);
    padf_[i] = static_cast<Dtype>((poolf_[i] - 1.0)/2);
  }
}

template <typename Dtype>
void RandCatConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // set the top-layer to be nxc -- 
  vector<int> top_shape(2);
  top_shape[0] = N_*(bottom[start_id_]->num());

  // compute num-channels for the given bottom-data
  n_channels_ = 0;
  height_ = std::vector<int>(n_hblobs_);
  width_ = std::vector<int>(n_hblobs_);
  pixels_ = std::vector<int>(n_hblobs_);
  for (int i = 0; i < n_hblobs_; i++) {
	n_channels_ = n_channels_ + bottom[i]->channels();
	height_[i] = bottom[i]->height();
    	width_[i] = bottom[i]->width();
    	pixels_[i] = height_[i] * width_[i];
  }
  //LOG(INFO) << "NUM Channels: " << n_channels_;
  top_shape[1] = n_channels_;
  top[0]->Reshape(top_shape);

  // set the surface-norm layer to be nx3 --
  top_shape[1] = 3; // TODO - make it flexible by providing param
  top[1]->Reshape(top_shape);
}

template <typename Dtype>
void RandCatConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
				int x_pt = j_pt % (bottom[start_id_]->width());
				int y_pt = (int) j_pt/(bottom[start_id_]->width());

				//LOG(INFO) << "J_pt" << j_pt << " X_pt: " << x_pt << " Y_pt: " << y_pt;
				rand_points_.push_back(x_pt);
				rand_points_.push_back(y_pt);
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
		for (int j = 0; j < (bottom[start_id_]->height()*bottom[start_id_]->width()); j++) {
			int data_pt = valid_data[j];
                        if(data_pt == 1) {
				rand_points_.push_back(i);
                        	int x_pt = j % (bottom[start_id_]->width());
                        	int y_pt = (int) j/(bottom[start_id_]->width());
                        	rand_points_.push_back(x_pt);
                        	rand_points_.push_back(y_pt);
			}
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
  int n, ch, tx1, tx2, ty1, ty2;
  Dtype tx, ty, rx, ry;
  Dtype x_pt, y_pt;
  int s = 0; int s_sn = 0;
  for(; i < N_*bottom_nums; i++) {
	n = int(rand_points_[j++]);
	x_pt = int(rand_points_[j++]);
	y_pt = int(rand_points_[j++]);
        // then find the corresponding locations
   	for (int b = 0; b < n_hblobs_; b++) {
      		tx = (x_pt-padf_[b])/poolf_[b];
     		ty = (y_pt-padf_[b])/poolf_[b];
      		tx1 = static_cast<int>(floor(tx));
      		ty1 = static_cast<int>(floor(ty));
      		tx2 = static_cast<int>(ceil(tx));
      		ty2 = static_cast<int>(ceil(ty));
      		// check if they are within the size limit
      		// CHECK_GE(tx1, 0);
      		tx1 = tx1 > 0 ? tx1 : 0;
      		tx2 = tx2 > 0 ? tx2 : 0;
		//if(tx2 == width_[b]){
		//	tx2 = tx1;
		//}
		
      		CHECK_LT(tx2, width_[b]);
      		// CHECK_GE(ty1, 0);
      		ty1 = ty1 > 0 ? ty1 : 0;
      		ty2 = ty2 > 0 ? ty2 : 0;
		//if(ty2 == height_[b]){
		//	ty2 = ty1;
		//}

		//LOG(INFO) << "TY1" << ty1 << "TY2: " << ty2 << " Height: " << height_[b]; 
     		//CHECK_LT(ty2, height_[b]);
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

	// and accumulate the surface normals (hard-coded for surface normals)
	for(int bc = 0; bc < 3; bc++){
		int init_sn = n*3*bottom_width*bottom_height + 
			      bc*bottom_width*bottom_width + y_pt*bottom_width + x_pt;
		top_sn[s_sn++] = sn_data[init_sn];
	}
  }
}

template <typename Dtype>
void RandCatConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 
    if(propagate_down[0]) {
	   //
	  const int bottom_nums = bottom[start_id_]->num();
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
  	  int n, ch, tx1, tx2, ty1, ty2;
 	  Dtype tx, ty, rx, ry;
  	  Dtype x_pt, y_pt;
  	  int s = 0; 
  	  for(; i < N_*bottom_nums; i++) {
        	n = int(rand_points_[j++]);
        	x_pt = int(rand_points_[j++]);
        	y_pt = int(rand_points_[j++]);
        	// then find the corresponding locations
        	for (int b = 0; b < n_hblobs_; b++) {
                	tx = (x_pt-padf_[b])/poolf_[b];
               	 	ty = (y_pt-padf_[b])/poolf_[b];
                	tx1 = static_cast<int>(floor(tx));
                	ty1 = static_cast<int>(floor(ty));
                	tx2 = static_cast<int>(ceil(tx));
                	ty2 = static_cast<int>(ceil(ty));
                	// check if they are within the size limit
                	// CHECK_GE(tx1, 0);
                	tx1 = tx1 > 0 ? tx1 : 0;
                	tx2 = tx2 > 0 ? tx2 : 0;
                	//if(tx2 == width_[b]){
                        //	tx2 = tx1;
                	//}
                	CHECK_LT(tx2, width_[b]);

                	// CHECK_GE(ty1, 0);
                	ty1 = ty1 > 0 ? ty1 : 0;
                	ty2 = ty2 > 0 ? ty2 : 0;
                	//if(ty2 == height_[b]){
                        //	ty2 = ty1;
                	//}

                	//CHECK_LT(ty2, height_[b]);
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

INSTANTIATE_CLASS(RandCatConvLayer);
REGISTER_LAYER_CLASS(RandCatConv);

}  // namespace caffe
